pub mod activation;
pub mod linear;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod train;

use crate::autograd::gpu_context::GpuContext;
use crate::autograd::variable::Variable;
use activation::{ReLU, Softmax};
use linear::Linear;

use std::error::Error;
use std::sync::Arc;

use tokio::sync::Mutex;

pub enum Layer {
    Linear(Linear),
    ReLU(ReLU),
    Softmax(Softmax),
}

impl Layer {
    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        let output = match self {
            Layer::Linear(layer) => layer.forward(input).await?,
            Layer::ReLU(layer) => layer.forward(input).await?,
            Layer::Softmax(layer) => layer.forward(input).await?,
        };

        if output.grad_fn.is_some() {
            println!(
                "  Layer output var {} has grad_fn, connected to input var {}",
                output.id, input.id
            );
        }

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        match self {
            Layer::Linear(layer) => layer.parameters(),
            Layer::ReLU(layer) => layer.parameters(),
            Layer::Softmax(layer) => layer.parameters(),
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        match self {
            Layer::Linear(layer) => layer.parameters_mut(),
            Layer::ReLU(layer) => layer.parameters_mut(),
            Layer::Softmax(layer) => layer.parameters_mut(),
        }
    }
}

pub mod layers {
    use super::*;

    pub async fn linear(
        in_features: usize,
        out_features: usize,
        context: Arc<Mutex<GpuContext>>,
    ) -> Result<Layer, Box<dyn Error>> {
        return Ok(Layer::Linear(
            Linear::new(in_features, out_features, true, context).await?,
        ));
    }

    pub fn relu() -> Layer {
        return Layer::ReLU(ReLU::new());
    }

    pub fn softmax() -> Layer {
        return Layer::Softmax(Softmax::new());
    }
}
