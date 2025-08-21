use super::loss::CrossEntropyLoss;
use super::model::Sequential;
use super::optimizer::SGD;
use super::{GpuContext, Variable};
use crate::autograd::{COMPUTATION_GRAPH, GRADIENT_REGISTRY};

use std::sync::Arc;

use tokio::sync::Mutex;

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
        println!("📍 Forward pass starting...");

        COMPUTATION_GRAPH.clear().await;
        GRADIENT_REGISTRY.clear_gradients().await;

        let predictions = self.model.forward(inputs).await?;

        println!("📍 Loss calculation...");
        let mut loss = self.loss_fn.forward(&predictions, targets).await?;
        let loss_value = loss.tensor.data[0];

        println!("📍 Backward pass starting...");
        {
            let context = self.context.lock().await;
            context.backward(&mut loss).await?;
        }

        println!("📍 Checking parameter gradients...");
        for (i, param) in self.model.parameters().iter().enumerate() {
            if let Some(grad) = GRADIENT_REGISTRY.get_gradient(param.id).await {
                println!(
                    "✅ Parameter {} (var {}): gradient norm {:.6}",
                    i,
                    param.id,
                    grad.data.iter().map(|x| x * x).sum::<f32>().sqrt()
                );
            } else {
                println!("❌ No gradient for parameter {} (var {})", i, param.id);
            }
        }

        println!("📍 Optimizer step...");
        let mut params = self.model.parameters_mut();
        self.optimizer.step(&mut params).await;
        self.optimizer.zero_grad(&mut params).await;

        Ok(loss_value)
    }
}
