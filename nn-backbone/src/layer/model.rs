use super::{Layer, Variable};

use std::error::Error;

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub fn new() -> Self {
        return Self { layers: Vec::new() };
    }

    pub fn add(mut self, layer: Layer) -> Self {
        self.layers.push(layer);

        return self;
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn len(&self) -> usize {
        return self.layers.len();
    }

    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        let mut current = input.clone();

        for (_, layer) in self.layers.iter_mut().enumerate() {
            current = layer.forward(&current).await?;
        }

        return Ok(current);
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        return self
            .layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect();
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return self
            .layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect();
    }
}

/*
desired API:

let mut model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU::new())
    .add(Linear::new(128, 10));

let output = model.forward(&input);
let loss = CrossEntropyLoss::new().forward(&output, &targets);
loss.backward();
*/
