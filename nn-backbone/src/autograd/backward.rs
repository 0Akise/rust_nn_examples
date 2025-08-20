use super::{BackwardFn, GradientComputer, GradientTask};

use gpu_accel::{Shape, Tensor};

use std::sync::Arc;

pub struct BackwardAdd {
    pub input_a_id: usize,
    pub input_b_id: usize,
}

impl BackwardFn for BackwardAdd {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        computer.queue_gradient(GradientTask::Add {
            grad_output: grad_output.clone(),
            target_var_id: self.input_a_id,
        });

        computer.queue_gradient(GradientTask::Add {
            grad_output: grad_output.clone(),
            target_var_id: self.input_b_id,
        });
    }
}

pub struct BackwardMul {
    pub input_a_id: usize,
    pub input_a_tensor: Arc<Vec<f32>>,
    pub input_a_shape: Shape,
    pub input_b_id: usize,
    pub input_b_tensor: Arc<Vec<f32>>,
    pub input_b_shape: Shape,
}

impl BackwardFn for BackwardMul {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        computer.queue_gradient(GradientTask::Mul {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_b_tensor.clone(),
            other_tensor_shape: self.input_b_shape.clone(),
            target_var_id: self.input_a_id,
        });

        computer.queue_gradient(GradientTask::Mul {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_a_tensor.clone(),
            other_tensor_shape: self.input_a_shape.clone(),
            target_var_id: self.input_b_id,
        });
    }
}

pub struct BackwardMatMul {
    pub input_a_id: usize,
    pub input_a_tensor: Arc<Vec<f32>>,
    pub input_a_shape: Shape,
    pub input_b_id: usize,
    pub input_b_tensor: Arc<Vec<f32>>,
    pub input_b_shape: Shape,
}

impl BackwardFn for BackwardMatMul {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        println!(
            "    BackwardMatMul: Computing gradients for vars {} and {}",
            self.input_a_id, self.input_b_id
        );

        computer.queue_gradient(GradientTask::MatMul {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_b_tensor.clone(),
            other_tensor_shape: self.input_b_shape.clone(),
            is_first_input: true,
            target_var_id: self.input_a_id,
        });

        computer.queue_gradient(GradientTask::MatMul {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_a_tensor.clone(),
            other_tensor_shape: self.input_a_shape.clone(),
            is_first_input: false,
            target_var_id: self.input_b_id,
        });
    }
}

pub struct BackwardDot {
    pub input_a_id: usize,
    pub input_a_tensor: Arc<Vec<f32>>,
    pub input_a_shape: Shape,
    pub input_b_id: usize,
    pub input_b_tensor: Arc<Vec<f32>>,
    pub input_b_shape: Shape,
}

impl BackwardFn for BackwardDot {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        computer.queue_gradient(GradientTask::Dot {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_b_tensor.clone(),
            other_tensor_shape: self.input_b_shape.clone(),
            target_var_id: self.input_a_id,
        });

        computer.queue_gradient(GradientTask::Dot {
            grad_output: grad_output.clone(),
            other_tensor_data: self.input_a_tensor.clone(),
            other_tensor_shape: self.input_a_shape.clone(),
            target_var_id: self.input_b_id,
        });
    }
}

pub struct BackwardTranspose {
    pub input_id: usize,
}

impl BackwardFn for BackwardTranspose {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        computer.queue_gradient(GradientTask::Transpose {
            grad_output: grad_output.clone(),
            target_var_id: self.input_id,
        });
    }
}

pub struct BackwardReLU {
    pub input_id: usize,
    pub input_data: Arc<Vec<f32>>,
}

impl BackwardFn for BackwardReLU {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        let grad_input: Vec<f32> = self
            .input_data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();

        computer.queue_gradient(GradientTask::Add {
            grad_output: Tensor::new(grad_input, grad_output.shape.clone()),
            target_var_id: self.input_id,
        });
    }
}

pub struct BackwardSoftmax {
    pub input_id: usize,
    pub output_data: Arc<Vec<f32>>,
}

impl BackwardFn for BackwardSoftmax {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        println!(
            "    BackwardSoftmax: Computing gradient for var {}",
            self.input_id
        );

        // Jacobian computation...
        let dot_product: f32 = self
            .output_data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(s, g)| s * g)
            .sum();

        let grad_input: Vec<f32> = self
            .output_data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(s, g)| s * (g - dot_product))
            .collect();

        println!(
            "    BackwardSoftmax: Queueing gradient for var {}",
            self.input_id
        );
        computer.queue_gradient(GradientTask::Add {
            grad_output: Tensor::new(grad_input, grad_output.shape.clone()),
            target_var_id: self.input_id,
        });
    }
}

pub struct BackwardCrossEntropy {
    pub predictions_id: usize,
    pub predictions_data: Arc<Vec<f32>>,
    pub targets_data: Arc<Vec<f32>>,
    pub batch_size: usize,
}

impl BackwardFn for BackwardCrossEntropy {
    fn backward(&self, grad_output: &Tensor, computer: &GradientComputer) {
        let mut grad_data = Vec::with_capacity(self.predictions_data.len());

        for (pred, target) in self.predictions_data.iter().zip(self.targets_data.iter()) {
            let grad = -target / pred.max(1e-15) / self.batch_size as f32;
            grad_data.push(grad * grad_output.data[0]);
        }

        let gradient = Tensor::new(
            grad_data,
            gpu_accel::Shape::new(vec![
                self.batch_size,
                self.predictions_data.len() / self.batch_size,
            ]),
        );

        computer.queue_gradient(GradientTask::Add {
            grad_output: gradient,
            target_var_id: self.predictions_id,
        });
    }
}
