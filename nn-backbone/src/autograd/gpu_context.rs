use crate::autograd::variable::Variable;
use crate::autograd::GradientComputer;
use crate::autograd::{COMPUTATION_GRAPH, GRADIENT_REGISTRY};

use gpu_accel::gpu_module::GpuModule;
use gpu_accel::Tensor;

use std::error::Error;
use std::sync::Arc;

use tokio::sync::Mutex;

pub struct GpuContext {
    gpu: Arc<Mutex<GpuModule>>,
    computer: GradientComputer,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        println!("Initializing GPU... 🌌");

        let gpu = Arc::new(Mutex::new(GpuModule::new().await?));
        let computer = GradientComputer::new(gpu.clone()).await?;

        return Ok(Self { gpu, computer });
    }

    pub fn gpu(&self) -> &Arc<Mutex<GpuModule>> {
        return &self.gpu;
    }

    pub fn computer(&self) -> &GradientComputer {
        return &self.computer;
    }

    pub async fn forward_add(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.add(b, &self.gpu).await;
    }

    pub async fn forward_mul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.mul(b, &self.gpu).await;
    }

    pub async fn forward_matmul(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.matmul(b, &self.gpu).await;
    }

    pub async fn forward_dot(
        &self,
        a: &Variable,
        b: &Variable,
    ) -> Result<Variable, Box<dyn Error>> {
        return a.dot(b, &self.gpu).await;
    }

    pub async fn forward_transpose(&self, a: &Variable) -> Result<Variable, Box<dyn Error>> {
        return a.transpose(&self.gpu).await;
    }

    pub async fn backward(&self, loss: &mut Variable) -> Result<(), Box<dyn Error>> {
        println!("🎯 Starting full graph backward propagation");

        loss.grad = Some(Tensor::ones(loss.tensor.shape.clone()));

        GRADIENT_REGISTRY
            .store_gradient(loss.id, loss.grad.as_ref().unwrap().clone())
            .await;

        let backward_order = COMPUTATION_GRAPH.get_backward_order(loss.id).await;

        println!("  Backward order: {:?}", backward_order);

        if backward_order.len() <= 1 {
            println!(
                "  ⚠️ WARNING: Graph has {} nodes - loss may not be connected!",
                backward_order.len()
            );
        }

        let mut tasks_queued = 0;

        for (idx, var_id) in backward_order.iter().enumerate() {
            if let Some(node) = COMPUTATION_GRAPH.get_node(*var_id).await {
                if let Some(grad) = GRADIENT_REGISTRY.get_gradient(*var_id).await {
                    println!(
                        "  Processing var {} with gradient ({}/{})",
                        var_id,
                        idx + 1,
                        backward_order.len()
                    );

                    if let Some(grad_fn) = &node.grad_fn {
                        println!("    Executing grad_fn for var {}", var_id);

                        grad_fn.backward(&grad, self.computer());
                        tasks_queued += 1;

                        if tasks_queued % 4 == 0 {
                            println!("    ⏸️ Pausing to let gradient computer catch up...");

                            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                            let gpu = self.gpu.lock().await;
                            let _ = gpu.device.poll(wgpu::PollType::Wait);
                            drop(gpu);

                            println!("    ▶️ Continuing...");
                        }
                    }
                }
            }
        }

        println!("  ⏸️ Waiting for all gradient computations to complete...");
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let gpu = self.gpu.lock().await;
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        drop(gpu);

        println!("  ✅ Backward pass complete");

        Ok(())
    }

    pub async fn gpu_info(&self) -> String {
        let gpu = self.gpu.lock().await;

        format!("{}", gpu.info.name)
    }
}
