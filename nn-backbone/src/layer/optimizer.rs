use crate::autograd::variable::Variable;

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub async fn step(&self, parameters: &mut [&mut Variable]) {
        for param in parameters {
            if let Some(grad) = param.get_computed_gradient().await {
                println!(
                    "📊 Applying gradient with norm: {:.6}",
                    grad.data.iter().map(|x| x * x).sum::<f32>().sqrt()
                );

                let new_data: Vec<f32> = param
                    .tensor
                    .data
                    .iter()
                    .zip(grad.data.iter())
                    .map(|(p, g)| p - self.learning_rate * g)
                    .collect();

                param.tensor = gpu_accel::Tensor::new(new_data, param.tensor.shape.clone());
            } else {
                println!("⚠️  No gradient found for parameter {}", param.id);
            }
        }
    }

    pub async fn zero_grad(&self, parameters: &mut [&mut Variable]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}
