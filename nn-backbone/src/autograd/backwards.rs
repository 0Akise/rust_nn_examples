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
