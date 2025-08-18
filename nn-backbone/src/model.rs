use super::autograd::Variable;

pub trait Model {
    fn forward(&mut self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<&Variable>;
}

/*
desired API:

let mut model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU::new())
    .add(Linear::new(128, 10));

let output = model.forward(&input);
let loss = CrossEntropyLoss::new().forward(&output, &targets);
loss.backward();
*/
