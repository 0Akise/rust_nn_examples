pub mod backwards;
pub mod test;

use backwards::{BackwardAdd, BackwardMatMul, BackwardMul};

use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use gpu_accel::{GpuSession, Shape, Tensor};

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor);
}

pub struct Variable {
    pub data: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<dyn BackwardFn>>,
}

impl Variable {
    pub fn new(data: Tensor, requires_grad: bool) -> Self {
        return Self {
            data,
            grad: None,
            requires_grad,
            grad_fn: None,
        };
    }

    pub fn from_tensor(data: Tensor) -> Self {
        return Self::new(data, false);
    }

    pub fn with_grad(data: Tensor) -> Self {
        return Self::new(data, true);
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(Tensor::zeros(self.data.shape.clone()));
        }
    }

    pub fn shape(&self) -> &Shape {
        return &self.data.shape;
    }

    pub fn data(&self) -> &Tensor {
        return &self.data;
    }

    pub fn backward(&mut self) {
        if !self.requires_grad {
            panic!("Cannot call backward on Variable that doesn't require gradients");
        }

        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.data.shape.clone()));
        }

        if let Some(grad_fn) = &self.grad_fn {
            let grad_output = self.grad.as_ref().unwrap();

            grad_fn.backward(grad_output);
        }
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Self {
            data: Tensor::new(self.data.data.clone(), self.data.shape.clone()),
            grad: self
                .grad
                .as_ref()
                .map(|g| Tensor::new(g.data.clone(), g.shape.clone())),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        };
    }
}

impl Debug for Variable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return formatter
            .debug_struct("Variable")
            .field("data_shape", &self.data.shape.dims)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish();
    }
}

impl Variable {
    pub async fn add(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();

            session.add(&self.data, &other.data).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardAdd {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                session: session.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn mul(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();
            session.multiply(&self.data, &other.data).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardMul {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                input_a_data: Tensor::new(self.data.data.clone(), self.data.shape.clone()),
                input_b_data: Tensor::new(other.data.data.clone(), other.data.shape.clone()),
                session: session.clone(),
            }));
        }

        Ok(result)
    }

    pub async fn matmul(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();
            session.matmul(&self.data, &other.data).await?
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardMatMul {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                input_a_data: Tensor::new(self.data.data.clone(), self.data.shape.clone()),
                input_b_data: Tensor::new(other.data.data.clone(), other.data.shape.clone()),
                session: session.clone(),
            }));
        }

        Ok(result)
    }
}
