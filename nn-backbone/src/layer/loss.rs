use super::Variable;

use gpu_accel::Tensor;

use std::error::Error;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        return Self;
    }

    pub async fn forward(
        &self,
        predictions: &Variable,
        targets: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        let data_pred = &predictions.tensor.data;
        let data_target = &targets.tensor.data;
        let loss = data_pred
            .iter()
            .zip(data_target.iter())
            .map(|(pred, target)| -target * pred.max(1e-15).ln())
            .sum::<f32>()
            / (predictions.tensor.shape.dims[0] as f32);

        return Ok(Variable::with_grad(Tensor::new(
            vec![loss],
            gpu_accel::Shape::new(vec![1]),
        )));
    }
}
