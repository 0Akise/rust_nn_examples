pub mod activation;
pub mod linear;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod train;

use super::autograd::{GpuContext, Variable};
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
        match self {
            Layer::Linear(layer) => layer.forward(input).await,
            Layer::ReLU(layer) => layer.forward(input).await,
            Layer::Softmax(layer) => layer.forward(input).await,
        }
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
