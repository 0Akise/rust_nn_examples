use super::backward::{BackwardAdd, BackwardDot, BackwardMatMul, BackwardMul, BackwardTranspose};
use super::{BackwardFn, GradientComputer, GraphNode};
use super::{COMPUTATION_GRAPH, GRADIENT_REGISTRY, VARIABLE_ID_COUNTER};

use gpu_accel::gpu_module::GpuModule;
use gpu_accel::{Shape, Tensor};

use std::error::Error;
use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::Mutex;

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

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
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
