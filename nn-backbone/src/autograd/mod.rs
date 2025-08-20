pub mod backward;

use backward::{BackwardAdd, BackwardDot, BackwardMatMul, BackwardMul, BackwardTranspose};
use gpu_accel::gpu_module::GpuModule;
use gpu_accel::{Shape, Tensor};

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex, RwLock};
use wgpu;

static VARIABLE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer);
}

pub struct GradientRegistry {
    gradients: Arc<RwLock<HashMap<usize, Tensor>>>,
}

impl GradientRegistry {
    pub fn new() -> Self {
        Self {
            gradients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn store_gradient(&self, var_id: usize, new_gradient: Tensor) {
        let mut gradients = self.gradients.write().await;

        match gradients.get_mut(&var_id) {
            Some(existing_grad) => match existing_grad.add_inplace(&new_gradient) {
                Ok(()) => {}
                Err(e) => {
                    eprintln!("Gradient accumulation error for variable {}: {}", var_id, e);

                    *existing_grad = new_gradient;
                }
            },
            None => {
                gradients.insert(var_id, new_gradient);
            }
        }
    }

    pub async fn init_gradient_zero(&self, var_id: usize, shape: Shape) {
        let mut gradients = self.gradients.write().await;

        if !gradients.contains_key(&var_id) {
            gradients.insert(var_id, Tensor::zeros(shape));
        }
    }

    pub async fn has_gradient(&self, var_id: usize) -> bool {
        let gradients = self.gradients.read().await;

        gradients.contains_key(&var_id)
    }

    pub async fn get_gradient(&self, var_id: usize) -> Option<Tensor> {
        let gradients = self.gradients.read().await;
        gradients
            .get(&var_id)
            .map(|t| Tensor::new(t.data.to_vec(), t.shape.clone()))
    }

    pub async fn clear_gradients(&self) {
        let mut gradients = self.gradients.write().await;
        gradients.clear();
    }
}

#[derive(Clone)]
pub struct GraphNode {
    pub id: usize,
    pub grad_fn: Option<Arc<dyn BackwardFn>>,
    pub parents: Vec<usize>,
    pub requires_grad: bool,
}

pub struct ComputationGraph {
    nodes: Arc<RwLock<HashMap<usize, GraphNode>>>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register_node(&self, node: GraphNode) {
        let mut nodes = self.nodes.write().await;
        nodes.insert(node.id, node);
    }

    pub async fn get_node(&self, id: usize) -> Option<GraphNode> {
        let nodes = self.nodes.read().await;
        nodes.get(&id).cloned()
    }

    pub async fn clear(&self) {
        let mut nodes = self.nodes.write().await;
        nodes.clear();
    }

    pub async fn get_backward_order(&self, root_id: usize) -> Vec<usize> {
        let nodes = self.nodes.read().await;
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        fn dfs(
            id: usize,
            nodes: &HashMap<usize, GraphNode>,
            visited: &mut HashSet<usize>,
            order: &mut Vec<usize>,
        ) {
            if visited.contains(&id) {
                return;
            }
            visited.insert(id);

            if let Some(node) = nodes.get(&id) {
                for &parent_id in &node.parents {
                    dfs(parent_id, nodes, visited, order);
                }
            }

            order.push(id);
        }

        dfs(root_id, &nodes, &mut visited, &mut order);

        order.reverse();
        order
    }
}

lazy_static::lazy_static! {
    pub static ref COMPUTATION_GRAPH: ComputationGraph = ComputationGraph::new();
}

lazy_static::lazy_static! {
    pub static ref GRADIENT_REGISTRY: GradientRegistry = GradientRegistry::new();
}

#[derive(Debug, Clone)]
pub enum GradientTask {
    Add {
        grad_output: Tensor,
        target_var_id: usize,
    },
    Mul {
        grad_output: Tensor,
        other_tensor_data: Arc<Vec<f32>>,
        other_tensor_shape: Shape,
        target_var_id: usize,
    },
    MatMul {
        grad_output: Tensor,
        other_tensor_data: Arc<Vec<f32>>,
        other_tensor_shape: Shape,
        is_first_input: bool,
        target_var_id: usize,
    },
    Dot {
        grad_output: Tensor,
        other_tensor_data: Arc<Vec<f32>>,
        other_tensor_shape: Shape,
        target_var_id: usize,
    },
    Transpose {
        grad_output: Tensor,
        target_var_id: usize,
    },
}

pub struct GradientComputer {
    task_sender: mpsc::UnboundedSender<GradientTask>,
    shutdown_sender: Option<tokio::sync::oneshot::Sender<()>>,
}

impl GradientComputer {
    pub async fn new(session: Arc<Mutex<GpuModule>>) -> Result<Self, Box<dyn Error>> {
        let (task_sender, mut task_receiver) = mpsc::unbounded_channel::<GradientTask>();
        let (shutdown_sender, mut shutdown_receiver) = tokio::sync::oneshot::channel();
        let session_clone = session.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(task) = task_receiver.recv() => {
                        if let Err(e) = Self::process_gradient_task(task, &session_clone).await {
                            eprintln!("Gradient computation error: {}", e);
                        }
                    }
                    _ = &mut shutdown_receiver => {
                        println!("Gradient computer shutting down");
                        break;
                    }
                }
            }
        });

        Ok(Self {
            task_sender,
            shutdown_sender: Some(shutdown_sender),
        })
    }

    pub fn shutdown(&mut self) {
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(());
        }
    }

    async fn process_gradient_task(
        task: GradientTask,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<(), Box<dyn Error>> {
        println!(
            "🔄 Processing gradient task: {:?}",
            match &task {
                GradientTask::Add { target_var_id, .. } => format!("Add for var {}", target_var_id),
                GradientTask::Mul { target_var_id, .. } => format!("Mul for var {}", target_var_id),
                GradientTask::MatMul { target_var_id, .. } =>
                    format!("MatMul for var {}", target_var_id),
                GradientTask::Dot { target_var_id, .. } => format!("Dot for var {}", target_var_id),
                GradientTask::Transpose { target_var_id, .. } =>
                    format!("Transpose for var {}", target_var_id),
            }
        );

        let mut session = session.lock().await;

        match task {
            GradientTask::Add {
                grad_output,
                target_var_id,
            } => {
                println!("  Storing gradient for var {}", target_var_id);
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, grad_output)
                    .await;
                println!("  ✅ Gradient stored for var {}", target_var_id);
            }

            GradientTask::Mul {
                grad_output,
                other_tensor_data,
                other_tensor_shape,
                target_var_id,
            } => {
                let other_tensor = Tensor {
                    data: other_tensor_data,
                    shape: other_tensor_shape,
                };
                let gradient = session.mul(&grad_output, &other_tensor).await?;
                println!("  Storing gradient for var {}", target_var_id);
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
                println!("  ✅ Gradient stored for var {}", target_var_id);
            }

            GradientTask::MatMul {
                grad_output,
                other_tensor_data,
                other_tensor_shape,
                is_first_input,
                target_var_id,
            } => {
                let other_tensor = Tensor {
                    data: other_tensor_data,
                    shape: other_tensor_shape,
                };

                let gradient = if is_first_input {
                    let b_transposed = session.transpose(&other_tensor).await?;
                    session.matmul(&grad_output, &b_transposed).await?
                } else {
                    let a_transposed = session.transpose(&other_tensor).await?;
                    session.matmul(&a_transposed, &grad_output).await?
                };
                println!("  Storing gradient for var {}", target_var_id);
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
                println!("  ✅ Gradient stored for var {}", target_var_id);
            }

            GradientTask::Dot {
                grad_output,
                other_tensor_data,
                other_tensor_shape,
                target_var_id,
            } => {
                let other_tensor = Tensor {
                    data: other_tensor_data,
                    shape: other_tensor_shape,
                };

                let scalar_value = grad_output.data[0];
                let expanded_scalar = Tensor::new(
                    vec![scalar_value; other_tensor.shape.total_elements()],
                    other_tensor.shape.clone(),
                );
                let gradient = session.mul(&expanded_scalar, &other_tensor).await?;
                println!("  Storing gradient for var {}", target_var_id);
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
                println!("  ✅ Gradient stored for var {}", target_var_id);
            }

            GradientTask::Transpose {
                grad_output,
                target_var_id,
            } => {
                let gradient = session.transpose(&grad_output).await?;
                println!("  Storing gradient for var {}", target_var_id);
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
                println!("  ✅ Gradient stored for var {}", target_var_id);
            }
        }

        return Ok(());
    }

    pub fn queue_gradient(&self, task: GradientTask) {
        println!("📨 Queueing gradient task");

        if let Err(_) = self.task_sender.send(task) {
            eprintln!("❌ Failed to queue gradient computation - channel closed!");
        }
    }
}

pub struct GpuContext {
    gpu: Arc<Mutex<GpuModule>>,
    computer: GradientComputer,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        println!("Initializing GPU... 🌌");

        let gpu = Arc::new(Mutex::new(GpuModule::new().await?));
        let computer = GradientComputer::new(gpu.clone()).await?;

        return Ok(Self { gpu, computer });
    }

    pub async fn reset(&mut self) -> Result<(), Box<dyn Error>> {
        println!("🔄 Initiating GPU context reset...");

        self.computer.shutdown();

        GRADIENT_REGISTRY.clear_gradients().await;

        {
            let gpu = self.gpu.lock().await;
            let _ = gpu.device.poll(wgpu::PollType::Wait);
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let new_gpu = Arc::new(Mutex::new(GpuModule::new().await?));

        self.computer = GradientComputer::new(new_gpu.clone()).await?;
        self.gpu = new_gpu;

        println!("✅ GPU context reset complete");

        Ok(())
    }

    pub fn gpu(&self) -> &Arc<Mutex<GpuModule>> {
        return &self.gpu;
    }

    pub fn computer(&self) -> &GradientComputer {
        return &self.computer;
    }

    pub async fn forward_add(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.add(b, &self.gpu).await;
    }

    pub async fn forward_mul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.mul(b, &self.gpu).await;
    }

    pub async fn forward_matmul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.matmul(b, &self.gpu).await;
    }

    pub async fn forward_dot(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.dot(b, &self.gpu).await;
    }

    pub async fn forward_transpose(&self, a: &Variable) -> Result<Variable, Box<dyn Error>> {
        return a.transpose(&self.gpu).await;
    }

    pub async fn backward(&self, loss: &mut Variable) -> Result<(), Box<dyn Error>> {
        println!("🎯 Starting full graph backward propagation");

        loss.grad = Some(Tensor::ones(loss.tensor.shape.clone()));

        GRADIENT_REGISTRY
            .store_gradient(loss.id, loss.grad.as_ref().unwrap().clone())
            .await;

        let backward_order = COMPUTATION_GRAPH.get_backward_order(loss.id).await;

        println!("  Backward order: {:?}", backward_order);

        for var_id in backward_order {
            if let Some(node) = COMPUTATION_GRAPH.get_node(var_id).await {
                if let Some(grad) = GRADIENT_REGISTRY.get_gradient(var_id).await {
                    println!("  Processing var {} with gradient", var_id);

                    if let Some(grad_fn) = &node.grad_fn {
                        println!("    Executing grad_fn for var {}", var_id);

                        grad_fn.backward(&grad, self.computer());

                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        Ok(())
    }

    pub async fn gpu_info(&self) -> String {
        let gpu = self.gpu.lock().await;

        format!("{}", gpu.info.name)
    }
}

pub struct Variable {
    pub id: usize,
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<dyn BackwardFn>>,
    pub parents: Vec<usize>,
}

impl Variable {
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        let id = VARIABLE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            id,
            tensor,
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: Vec::new(),
        }
    }

    pub async fn register_in_graph(&self) {
        if self.requires_grad || self.grad_fn.is_some() {
            let node = GraphNode {
                id: self.id,
                grad_fn: self.grad_fn.clone(),
                parents: self.parents.clone(),
                requires_grad: self.requires_grad,
            };
            COMPUTATION_GRAPH.register_node(node).await;
        }
    }

    pub async fn create_result(
        data: Tensor,
        parents: Vec<usize>,
        grad_fn: Option<Arc<dyn BackwardFn>>,
    ) -> Self {
        let requires_grad = !parents.is_empty();

        let mut result = Self::new(data, requires_grad);
        result.parents = parents;
        result.grad_fn = grad_fn;
        result.register_in_graph().await;

        result
    }

    pub fn from_tensor(tensor: Tensor) -> Self {
        return Self::new(tensor, false);
    }

    pub fn with_grad(tensor: Tensor) -> Self {
        return Self::new(tensor, true);
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Tensor::zeros(self.tensor.shape.clone()));
        }
    }

    pub fn shape(&self) -> &Shape {
        return &self.tensor.shape;
    }

    pub fn tensor(&self) -> &Tensor {
        return &self.tensor;
    }

    pub async fn init_gradient(&self) {
        if self.requires_grad {
            GRADIENT_REGISTRY
                .init_gradient_zero(self.id, self.tensor.shape.clone())
                .await;
        }
    }

    pub async fn has_computed_gradient(&self) -> bool {
        GRADIENT_REGISTRY.has_gradient(self.id).await
    }

    pub async fn get_computed_gradient(&self) -> Option<Tensor> {
        GRADIENT_REGISTRY.get_gradient(self.id).await
    }

    pub async fn backward(&mut self, computer: &GradientComputer) {
        if !self.requires_grad {
            panic!("Cannot call backward on Variable that doesn't require gradients");
        }

        println!("🎯 Starting backward for variable {}", self.id);

        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.tensor.shape.clone()));
        }

        GRADIENT_REGISTRY
            .store_gradient(self.id, self.grad.as_ref().unwrap().clone())
            .await;

        println!("  Initial gradient stored for var {}", self.id);

        if let Some(grad_fn) = &self.grad_fn {
            println!("  Executing grad_fn for var {}", self.id);
            let grad_output = self.grad.as_ref().unwrap();
            grad_fn.backward(grad_output, computer);
        } else {
            println!("  No grad_fn for var {} (this is the loss/root)", self.id);
        }
    }

    pub async fn backward_full(&mut self, computer: &GradientComputer) {
        self.backward(computer).await;

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    pub async fn add(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.add(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Arc::new(BackwardAdd {
                input_a_id: self.id,
                input_b_id: other.id,
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let result = Variable::create_result(result_data, vec![self.id, other.id], grad_fn).await;

        Ok(result)
    }

    pub async fn mul(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.mul(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Arc::new(BackwardMul {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let result = Variable::create_result(result_data, vec![self.id, other.id], grad_fn).await;

        return Ok(result);
    }

    pub async fn matmul(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.matmul(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Arc::new(BackwardMatMul {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let result = Variable::create_result(result_data, vec![self.id, other.id], grad_fn).await;

        Ok(result)
    }

    pub async fn dot(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.dot(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Arc::new(BackwardDot {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let result = Variable::create_result(result_data, vec![self.id, other.id], grad_fn).await;

        return Ok(result);
    }

    pub async fn transpose(
        &self,
        session: &Arc<Mutex<GpuModule>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.transpose(&self.tensor).await?
        };

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Arc::new(BackwardTranspose { input_id: self.id }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let result = Variable::create_result(result_data, vec![self.id], grad_fn).await;

        return Ok(result);
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Self {
            id: self.id,
            tensor: self.tensor.clone(),
            grad: self.grad.as_ref().map(|g| g.clone()),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            parents: Vec::new(),
        };
    }
}

impl Debug for Variable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return formatter
            .debug_struct("Variable")
            .field("data_shape", &self.tensor.shape.dims)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish();
    }
}
