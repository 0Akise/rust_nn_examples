use crate::autograd::gpu_context::GpuContext;
use crate::autograd::variable::Variable;

use gpu_accel::{Shape, Tensor};

use std::error::Error;
use std::sync::Arc;

use tokio::sync::Mutex;

pub struct Linear {
    w: Variable,
    b: Option<Variable>,
    features_in: usize,
    features_out: usize,
    ctx: Arc<Mutex<GpuContext>>,
}

impl Linear {
    pub async fn new(
        features_in: usize,
        features_out: usize,
        use_bias: bool,
        ctx: Arc<Mutex<GpuContext>>,
    ) -> Result<Self, Box<dyn Error>> {
        let scale = (2.0 / (features_in + features_out) as f32).sqrt();
        let weight_data: Vec<f32> = (0..features_in * features_out)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let w = Variable::with_grad(Tensor::new(
            weight_data,
            Shape::new(vec![features_in, features_out]),
        ));

        println!("Linear layer created with weight var ID: {}", w.id);

        let b = if use_bias {
            let bias = Variable::with_grad(Tensor::new(
                vec![0.0; features_out],
                Shape::new(vec![features_out]),
            ));
            bias.register_in_graph().await;
            Some(bias)
        } else {
            None
        };

        return Ok(Self {
            w,
            b,
            features_in,
            features_out,
            ctx,
        });
    }

    pub async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn Error>> {
        println!(
            "    Linear forward: input var {}, weight var {}",
            input.id, self.w.id
        );

        let context = self.ctx.lock().await;
        let linear_output = context.forward_matmul(input, &self.w).await?;

        println!("    Linear matmul output: var {}", linear_output.id);

        if let Some(ref bias) = self.b {
            println!("    Adding bias var {}", bias.id);

            let data_bias = &bias.tensor.data;
            let bias_expanded = Variable::with_grad(Tensor::new(
                data_bias
                    .iter()
                    .cycle()
                    .take(linear_output.tensor.data.len())
                    .cloned()
                    .collect(),
                linear_output.tensor.shape.clone(),
            ));

            let result = context.forward_add(&linear_output, &bias_expanded).await?;

            println!("    Linear final output: var {}", result.id);

            return Ok(result);
        } else {
            return Ok(linear_output);
        }
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        let mut params = vec![&self.w];

        if let Some(ref bias) = self.b {
            params.push(bias);
        }

        return params;
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        let mut params = vec![&mut self.w];

        if let Some(ref mut bias) = self.b {
            params.push(bias);
        }

        return params;
    }
}
