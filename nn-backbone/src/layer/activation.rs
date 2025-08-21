use crate::autograd::backward::{BackwardReLU, BackwardSoftmax};
use crate::autograd::variable::Variable;
use crate::autograd::BackwardFn;

use gpu_accel::Tensor;

use std::error::Error;
use std::sync::Arc;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        return Self;
    }

    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        let data_output: Vec<f32> = input
            .tensor
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        let grad_fn = if input.requires_grad {
            Some(Arc::new(BackwardReLU {
                input_id: input.id,
                input_data: input.tensor.data.clone(),
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let output = Variable::create_result(
            Tensor::new(data_output, input.tensor.shape.clone()),
            vec![input.id],
            grad_fn,
        )
        .await;

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        return vec![];
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return vec![];
    }
}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        return Self;
    }

    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        let data_input = &input.tensor.data;
        let val_max = data_input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let val_exp: Vec<f32> = data_input.iter().map(|&x| (x - val_max).exp()).collect();
        let sum_exp: f32 = val_exp.iter().sum();
        let data_output: Vec<f32> = val_exp.iter().map(|&x| x / sum_exp).collect();

        let grad_fn = if input.requires_grad {
            Some(Arc::new(BackwardSoftmax {
                input_id: input.id,
                output_data: data_output.clone().into(),
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let output = Variable::create_result(
            Tensor::new(data_output, input.tensor.shape.clone()),
            vec![input.id],
            grad_fn,
        )
        .await;

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        return vec![];
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return vec![];
    }
}
