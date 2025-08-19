use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::Variable;
use nn_backbone::expr::{ExprExecutor, ExprGraph};

pub async fn operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Expression Builder");

    let mut executor = ExprExecutor::new().await?;

    let input_var = Variable::with_grad(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![2, 2]),
    ));
    let weights_var = Variable::with_grad(Tensor::new(
        vec![0.5, 1.0, 1.5, 2.0],
        Shape::new(vec![2, 2]),
    ));
    let bias_var = Variable::with_grad(Tensor::new(
        vec![0.1, 0.2, 0.3, 0.4],
        Shape::new(vec![2, 2]),
    ));

    let mut graph = ExprGraph::new();
    let input = graph.input(input_var);
    let w = graph.input(weights_var);
    let b = graph.input(bias_var);

    let linear = graph.matmul(input, w)?;
    let with_b = graph.add(linear, b)?;
    let output = graph.relu(with_b)?;

    println!("✅ Expression graph built. Nodes: {}", graph.num_nodes());

    let result = executor.compute_with_grad(&graph, output).await?;

    println!("Result shape: {:?}", result.tensor.shape.dims);
    println!("Result data: {:?}", &result.tensor.data[0..4]);

    return Ok(());
}

#[tokio::main]
async fn main() {
    pollster::block_on(async {
        if let Err(e) = operations().await {
            println!("Error: {}", e);
        }
    });
}
