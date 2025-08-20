use super::autograd::{GpuContext, Variable};

use gpu_accel::{Shape, Tensor};

use std::sync::Arc;

use tokio::sync::Mutex;

pub struct Linear {
    w: Variable,
    b: Option<Variable>,
    features_in: usize,
    features_out: usize,
    ctx: Arc<Mutex<GpuContext>>,
    name: String,
}

impl Linear {
    pub async fn new(
        features_in: usize,
        features_out: usize,
        use_bias: bool,
        ctx: Arc<Mutex<GpuContext>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let scale = (2.0 / (features_in + features_out) as f32).sqrt();
        let weight_data: Vec<f32> = (0..features_in * features_out)
            .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let w = Variable::with_grad(Tensor::new(
            weight_data,
            Shape::new(vec![features_in, features_out]),
        ));
        let b = if use_bias {
            let bias_data = vec![0.0; features_out];
            Some(Variable::with_grad(Tensor::new(
                bias_data,
                Shape::new(vec![features_out]),
            )))
        } else {
            None
        };

        return Ok(Self {
            w,
            b,
            features_in,
            features_out,
            ctx,
            name: format!("Linear({} -> {})", features_in, features_out),
        });
    }
}

impl Linear {
    async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn std::error::Error>> {
        let context = self.ctx.lock().await;
        let linear_output = context.forward_matmul(input, &self.w).await?;
        if let Some(ref bias) = self.b {
            let bias_expanded = Variable::with_grad(Tensor::new(
                bias.tensor
                    .data
                    .iter()
                    .cycle()
                    .take(linear_output.tensor.data.len())
                    .cloned()
                    .collect(),
                linear_output.tensor.shape.clone(),
            ));

            context.forward_add(&linear_output, &bias_expanded).await
        } else {
            return Ok(linear_output);
        }
    }

    fn parameters(&self) -> Vec<&Variable> {
        let mut params = vec![&self.w];
        if let Some(ref bias) = self.b {
            params.push(bias);
        }

        return params;
    }

    fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        let mut params = vec![&mut self.w];
        if let Some(ref mut bias) = self.b {
            params.push(bias);
        }

        return params;
    }
}

pub struct ReLU {
    name: String,
}

impl ReLU {
    pub fn new() -> Self {
        return Self {
            name: "ReLU".to_string(),
        };
    }
}

impl ReLU {
    async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn std::error::Error>> {
        let relu_data: Vec<f32> = input
            .tensor
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        return Ok(Variable::with_grad(Tensor::new(
            relu_data,
            input.tensor.shape.clone(),
        )));
    }

    fn parameters(&self) -> Vec<&Variable> {
        return vec![];
    }

    fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return vec![];
    }
}

pub struct Softmax {
    name: String,
}

impl Softmax {
    pub fn new() -> Self {
        Self {
            name: "Softmax".to_string(),
        }
    }
}

impl Softmax {
    async fn forward(&mut self, input: &Variable) -> Result<Variable, Box<dyn std::error::Error>> {
        let input_data = &input.tensor.data;
        let mut output_data = Vec::with_capacity(input_data.len());
        let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f32> = input_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();

        for exp_val in exp_values {
            output_data.push(exp_val / sum_exp);
        }

        return Ok(Variable::with_grad(Tensor::new(
            output_data,
            input.tensor.shape.clone(),
        )));
    }

    fn parameters(&self) -> Vec<&Variable> {
        return vec![];
    }

    fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        return vec![];
    }
}

pub enum Layer {
    Linear(Linear),
    ReLU(ReLU),
    Softmax(Softmax),
}

impl Layer {
    pub async fn forward(
        &mut self,
        input: &Variable,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
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

    pub fn name(&self) -> &str {
        match self {
            Layer::Linear(layer) => &layer.name,
            Layer::ReLU(layer) => &layer.name,
            Layer::Softmax(layer) => &layer.name,
        }
    }
}

pub struct Sequential {
    layers: Vec<Layer>,
    name: String,
}

impl Sequential {
    pub fn new() -> Self {
        return Self {
            layers: Vec::new(),
            name: "Sequential".to_string(),
        };
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

    pub async fn forward(
        &mut self,
        input: &Variable,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let mut current = input.clone();

        for (_, layer) in self.layers.iter_mut().enumerate() {
            current = layer.forward(&current).await?;
        }

        return Ok(current);
    }

    pub fn parameters(&self) -> Vec<&Variable> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Variable> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    pub fn name(&self) -> &str {
        return &self.name;
    }
}

pub mod layers {
    use super::*;

    pub async fn linear(
        in_features: usize,
        out_features: usize,
        context: Arc<Mutex<GpuContext>>,
    ) -> Result<Layer, Box<dyn std::error::Error>> {
        Ok(Layer::Linear(
            Linear::new(in_features, out_features, true, context).await?,
        ))
    }

    pub fn relu() -> Layer {
        Layer::ReLU(ReLU::new())
    }

    pub fn softmax() -> Layer {
        Layer::Softmax(Softmax::new())
    }
}
