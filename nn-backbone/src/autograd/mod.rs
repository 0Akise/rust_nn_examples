use gpu_accel::Tensor;

pub struct Variable {
    pub data: Tensor,
    pub grad: Option<Tensor>,
    pub requires_grad: bool,
    pub grad_fn: Option<Box<dyn BackwardFn>>,
}

pub trait BackwardFn {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}
