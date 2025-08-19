use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::{GpuContext, Variable};

pub async fn operations() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new().await?;

    println!("\n🧪 Testing Simple Addition Autograd");

    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0], Shape::new(vec![2])));
    let b = Variable::with_grad(Tensor::new(vec![3.0, 4.0], Shape::new(vec![2])));

    println!("a: {:?}", a.tensor.data);
    println!("b: {:?}", b.tensor.data);

    let mut c = ctx.forward_add(&a, &b).await?;

    println!("c = a + b: {:?}", c.tensor.data);
    println!("\n🔄 Starting backward pass...");

    ctx.backward(&mut c);

    println!("✅ Backward pass completed!");
    println!("Variable c has grad_fn: {}", c.grad_fn.is_some());
    println!("\n🧪 Testing Matrix Multiplication Autograd");

    let a = Variable::with_grad(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![2, 2]),
    ));
    let b = Variable::with_grad(Tensor::new(
        vec![0.5, 1.0, 1.5, 2.0],
        Shape::new(vec![2, 2]),
    ));
    let mut c = ctx.forward_matmul(&a, &b).await?;

    println!("Matrix A: {:?}", a.tensor.data);
    println!("Matrix B: {:?}", b.tensor.data);
    println!("C = A @ B: {:?}", c.tensor.data);

    ctx.backward(&mut c);

    println!("✅ Matrix multiplication autograd completed!");
    println!("\n🧪 Testing Dot Product Autograd");

    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
    let b = Variable::with_grad(Tensor::new(vec![4.0, 5.0, 6.0], Shape::new(vec![3])));
    let mut c = ctx.forward_dot(&a, &b).await?;

    println!("C = A ‣ B: {:?}", c.tensor.data);
    println!("\n🔄 Starting backward pass...");

    ctx.backward(&mut c);

    println!("✅ Dot product autograd completed!");
    println!("\n🧪 Testing Transpose Autograd");

    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
    let a_t = ctx.forward_transpose(&a).await?;
    let b = Variable::with_grad(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![2, 2]),
    ));
    let mut b_t = ctx.forward_transpose(&b).await?;

    println!(
        "1D vector: {:?}, shape: {:?}",
        a.tensor.data, a.tensor.shape.dims
    );
    println!(
        "1D vector: {:?}, shape: {:?}",
        a_t.tensor.data, a_t.tensor.shape.dims
    );
    println!(
        "2D Matrix: {:?}, shape: {:?}",
        b.tensor.data, b.tensor.shape.dims
    );
    println!(
        "2D Matrix: {:?}, shape: {:?}",
        b_t.tensor.data, b_t.tensor.shape.dims
    );
    println!("\n🔄 Starting backward pass...");

    ctx.backward(&mut b_t);

    println!("✅ Transpose autograd completed!");

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
