use super::autograd::Variable;

use std::sync::{Arc, Mutex};

use gpu_accel::GpuSession;

pub trait Layer {
    fn forward(&mut self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<&Variable>;
}

pub struct Linear {
    weight: Variable,
    bias: Option<Variable>,
    session: Arc<Mutex<GpuSession>>,
}
