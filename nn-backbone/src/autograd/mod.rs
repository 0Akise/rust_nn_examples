pub mod backward;
pub mod gpu_context;
pub mod variable;

use gpu_accel::gpu_module::GpuModule;
use gpu_accel::{Shape, Tensor};

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex, RwLock};
use wgpu;

static VARIABLE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

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
            let mut pending_tasks = 0;

            loop {
                tokio::select! {
                    Some(task) = task_receiver.recv() => {
                        pending_tasks += 1;

                        if let Err(e) = Self::process_gradient_task(task, &session_clone).await {
                            eprintln!("Gradient computation error: {}", e);
                        }


                        if pending_tasks >= 4 {
                            println!("    🔄 GPU sync after {} gradient tasks", pending_tasks);
                            let session = session_clone.lock().await;
                            let _ = session.device.poll(wgpu::PollType::Wait);
                            drop(session);
                            pending_tasks = 0;


                            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
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

                let _ = session.device.poll(wgpu::PollType::Wait);

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

                    let _ = session.device.poll(wgpu::PollType::Wait);

                    session.matmul(&grad_output, &b_transposed).await?
                } else {
                    let a_transposed = session.transpose(&other_tensor).await?;

                    let _ = session.device.poll(wgpu::PollType::Wait);

                    session.matmul(&a_transposed, &grad_output).await?
                };

                let _ = session.device.poll(wgpu::PollType::Wait);

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

                let _ = session.device.poll(wgpu::PollType::Wait);

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

                let _ = session.device.poll(wgpu::PollType::Wait);

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

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer);
}
