use super::loss::CrossEntropyLoss;
use super::model::Sequential;
use super::optimizer::SGD;
use super::{GpuContext, Variable};

use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::time;

pub struct SequentialTrainer {
    pub model: Sequential,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
    context: Arc<Mutex<GpuContext>>,
}

impl SequentialTrainer {
    pub fn new(model: Sequential, learning_rate: f32, context: Arc<Mutex<GpuContext>>) -> Self {
        Self {
            model,
            optimizer: SGD::new(learning_rate),
            loss_fn: CrossEntropyLoss::new(),
            context,
        }
    }

    pub async fn train_step(
        &mut self,
        inputs: &Variable,
        targets: &Variable,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let predictions = self.model.forward(inputs).await?;
        let mut loss = self.loss_fn.forward(&predictions, targets).await?;
        let loss_value = loss.tensor.data[0];
        let context = self.context.lock().await;

        context.backward(&mut loss).await;

        drop(context);

        time::sleep(time::Duration::from_millis(100)).await;

        let mut params = self.model.parameters_mut();

        self.optimizer.step(&mut params).await;
        self.optimizer.zero_grad(&mut params).await;

        Ok(loss_value)
    }
}
