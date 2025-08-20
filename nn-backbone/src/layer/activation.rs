use super::Variable;

use gpu_accel::Tensor;

use std::error::Error;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        return Self;
    }

    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        let data_input = &input.tensor.data;
        let data_output = data_input
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        return Ok(Variable::with_grad(Tensor::new(
            data_output,
            input.tensor.shape.clone(),
        )));
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
        let mut data_output = Vec::with_capacity(data_input.len());

        for exp_val in val_exp {
            data_output.push(exp_val / sum_exp);
        }

        return Ok(Variable::with_grad(Tensor::new(
            data_output,
            input.tensor.shape.clone(),
        )));
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        return vec![];
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return vec![];
    }
}
