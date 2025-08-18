pub mod backwards;
pub mod test;

use backwards::{BackwardAdd, BackwardDot, BackwardMatMul, BackwardMul, BackwardTranspose};

use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use gpu_accel::{GpuSession, Shape, Tensor};

pub trait BackwardFn: Send + Sync {
    fn backward(&self, grad_output: &Tensor);
}

pub struct Variable {
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<dyn BackwardFn>>,
}

impl Variable {
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        return Self {
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

    pub fn backward(&mut self) {
        if !self.requires_grad {
            panic!("Cannot call backward on Variable that doesn't require gradients");
        }

        if self.grad.is_none() {
            self.grad = Some(Tensor::ones(self.tensor.shape.clone()));
        }

        if let Some(grad_fn) = &self.grad_fn {
            let grad_output = self.grad.as_ref().unwrap();

            grad_fn.backward(grad_output);
        }
    }

    pub async fn add(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();

            session.add(&self.tensor, &other.tensor).await.unwrap()
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

            session.mul(&self.tensor, &other.tensor).await.unwrap()
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardMul {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                input_a_data: Tensor::new(self.tensor.data.clone(), self.tensor.shape.clone()),
                input_b_data: Tensor::new(other.tensor.data.clone(), other.tensor.shape.clone()),
                session: session.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn matmul(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();

            session.matmul(&self.tensor, &other.tensor).await.unwrap()
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardMatMul {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                input_a_data: Tensor::new(self.tensor.data.clone(), self.tensor.shape.clone()),
                input_b_data: Tensor::new(other.tensor.data.clone(), other.tensor.shape.clone()),
                session: session.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn dot(
        &self,
        other: &Variable,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();

            session.dot(&self.tensor, &other.tensor).await.unwrap()
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));
            let other_ref = Arc::new(Mutex::new(other.clone()));

            result.grad_fn = Some(Arc::new(BackwardDot {
                input_a: Arc::downgrade(&self_ref),
                input_b: Arc::downgrade(&other_ref),
                input_a_data: Tensor::new(self.tensor.data.clone(), self.tensor.shape.clone()),
                input_b_data: Tensor::new(other.tensor.data.clone(), other.tensor.shape.clone()),
                session: session.clone(),
            }));
        }

        return Ok(result);
    }

    pub async fn transpose(
        &self,
        session: Arc<Mutex<GpuSession>>,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let result_data = {
            let mut session = session.lock().unwrap();

            session.transpose(&self.tensor).await.unwrap()
        };

        let requires_grad = self.requires_grad;
        let mut result = Variable::new(result_data, requires_grad);

        if requires_grad {
            let self_ref = Arc::new(Mutex::new(self.clone()));

            result.grad_fn = Some(Arc::new(BackwardTranspose {
                input: Arc::downgrade(&self_ref),
                session: session.clone(),
            }));
        }

        return Ok(result);
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Self {
            tensor: Tensor::new(self.tensor.data.clone(), self.tensor.shape.clone()),
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
            .field("data_shape", &self.tensor.shape.dims)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish();
    }
}
