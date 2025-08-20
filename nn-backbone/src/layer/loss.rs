use super::Variable;
use crate::autograd::backward::BackwardCrossEntropy;
use crate::autograd::BackwardFn;

use gpu_accel::{Shape, Tensor};

use std::error::Error;
use std::sync::Arc;

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

        let grad_fn = if predictions.requires_grad {
            Some(Arc::new(BackwardCrossEntropy {
                predictions_id: predictions.id,
                predictions_data: predictions.tensor.data.clone(),
                targets_data: targets.tensor.data.clone(),
                batch_size: predictions.tensor.shape.dims[0],
            }) as Arc<dyn BackwardFn>)
        } else {
            None
        };

        let loss_var = Variable::create_result(
            Tensor::new(vec![loss], Shape::new(vec![1])),
            vec![predictions.id],
            grad_fn,
        )
        .await;

        Ok(loss_var)
    }
}
