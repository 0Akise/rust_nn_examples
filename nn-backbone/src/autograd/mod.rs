pub mod backwards;

use backwards::{BackwardAdd, BackwardDot, BackwardMatMul, BackwardMul, BackwardTranspose};
use gpu_accel::{GpuSession, Shape, Tensor};

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex, RwLock};

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

lazy_static::lazy_static! {
    static ref GRADIENT_REGISTRY: GradientRegistry = GradientRegistry::new();
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
}

impl GradientComputer {
    pub async fn new(session: Arc<Mutex<GpuSession>>) -> Result<Self, Box<dyn Error>> {
        let (task_sender, mut task_receiver) = mpsc::unbounded_channel::<GradientTask>();
        let session_clone = session.clone();

        tokio::spawn(async move {
            while let Some(task) = task_receiver.recv().await {
                if let Err(e) = Self::process_gradient_task(task, &session_clone).await {
                    eprintln!("Gradient computation error: {}", e);
                }
            }
        });

        return Ok(Self { task_sender });
    }

    async fn process_gradient_task(
        task: GradientTask,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut session = session.lock().await;

        match task {
            GradientTask::Add {
                grad_output,
                target_var_id,
            } => {
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, grad_output)
                    .await;
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
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
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
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
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
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
            }

            GradientTask::Transpose {
                grad_output,
                target_var_id,
            } => {
                let gradient = session.transpose(&grad_output).await?;
                GRADIENT_REGISTRY
                    .store_gradient(target_var_id, gradient)
                    .await;
            }
        }

        return Ok(());
    }

    pub fn queue_gradient(&self, task: GradientTask) {
        if let Err(_) = self.task_sender.send(task) {
            eprintln!("Failed to queue gradient computation");
        }
    }
}

pub struct GpuContext {
    session: Arc<Mutex<GpuSession>>,
    computer: GradientComputer,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        println!("Initializing GPU... 🌌");

        let session = Arc::new(Mutex::new(GpuSession::new().await?));
        let computer = GradientComputer::new(session.clone()).await?;

        return Ok(Self { session, computer });
    }

    pub fn session(&self) -> &Arc<Mutex<GpuSession>> {
        return &self.session;
    }

    pub fn computer(&self) -> &GradientComputer {
        return &self.computer;
    }

    pub async fn forward_add(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.add(b, &self.session).await;
    }

    pub async fn forward_mul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.mul(b, &self.session).await;
    }

    pub async fn forward_matmul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.matmul(b, &self.session).await;
    }

    pub async fn forward_dot(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.dot(b, &self.session).await;
    }

    pub async fn forward_transpose(&self, a: &Variable) -> Result<Variable, Box<dyn Error>> {
        return a.transpose(&self.session).await;
    }

    pub fn backward(&self, var: &mut Variable) {
        var.backward(&self.computer)
    }

    pub async fn gpu_info(&self) -> String {
        let session = self.session.lock().await;

        format!("{}", session.gpu.info.name)
    }
}

pub struct Variable {
    pub id: usize,
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<dyn BackwardFn>>,
}

impl Variable {
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        return Self {
            id: VARIABLE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            tensor,
            grad: None,
            requires_grad,
            grad_fn: None,
        };
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

    pub fn backward(&mut self, computer: &GradientComputer) {
        if self.requires_grad != true {
            panic!("Cannot call backward on Variable that doesn't require gradients");
        }

        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.tensor.shape.clone()));
        }

        if let Some(grad_fn) = &self.grad_fn {
            let grad_output = self.grad.as_ref().unwrap();

            grad_fn.backward(grad_output, computer);
        }
    }

    pub async fn backward_async(&mut self, computer: &GradientComputer) {
        if self.requires_grad != true {
            panic!("Cannot call backward on Variable that doesn't require gradients");
        }

        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.tensor.shape.clone()));
        }

        self.init_gradient().await;

        if let Some(grad_fn) = &self.grad_fn {
            let grad_output = self.grad.as_ref().unwrap();

            grad_fn.backward(grad_output, computer);
        }
    }

    pub async fn add(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.add(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Arc::new(BackwardAdd {
                input_a_id: self.id,
                input_b_id: other.id,
            }));
        }

        Ok(result)
    }

    pub async fn mul(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;

            session.mul(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Arc::new(BackwardMul {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn matmul(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;
            session.matmul(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Arc::new(BackwardMatMul {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn dot(
        &self,
        other: &Variable,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;
            session.dot(&self.tensor, &other.tensor).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Arc::new(BackwardDot {
                input_a_id: self.id,
                input_a_tensor: self.tensor.data.clone(),
                input_a_shape: self.tensor.shape.clone(),
                input_b_id: other.id,
                input_b_tensor: other.tensor.data.clone(),
                input_b_shape: other.tensor.shape.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn transpose(
        &self,
        session: &Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn Error>> {
        let result_data = {
            let mut session = session.lock().await;
            session.transpose(&self.tensor).await?
        };

        let requires_grad = self.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Arc::new(BackwardTranspose { input_id: self.id }));
        }

        return Ok(result);
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Self {
            id: VARIABLE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            tensor: Tensor::new(self.tensor.data.to_vec(), self.tensor.shape.clone()),
            grad: self
                .grad
                .as_ref()
                .map(|g| Tensor::new(g.data.to_vec(), g.shape.clone())),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
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
