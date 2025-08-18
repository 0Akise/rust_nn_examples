use super::Variable;

use std::sync::{Arc, Mutex};

use gpu_accel::{GpuSession, Shape, Tensor};

pub async fn simple_addition() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Simple Addition Autograd");

    let session = Arc::new(Mutex::new(GpuSession::new().await?));
    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0], Shape::new(vec![2])));
    let b = Variable::with_grad(Tensor::new(vec![3.0, 4.0], Shape::new(vec![2])));

    println!("a: {:?}", a.data.data);
    println!("b: {:?}", b.data.data);

    let c = a.add(&b, session.clone()).await?;

    println!("c = a + b: {:?}", c.data.data);
    println!("\n🔄 Starting backward pass...");

    let mut c_mut = c;

    c_mut.backward();

    println!("✅ Backward pass completed!");
    println!("Variable c has grad_fn: {}", c_mut.grad_fn.is_some());

    return Ok(());
}

pub async fn compilation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing that everything compiles...");

    let session = Arc::new(Mutex::new(GpuSession::new().await?));
    let a = Variable::with_grad(Tensor::new(vec![1.0], Shape::new(vec![1])));
    let b = Variable::with_grad(Tensor::new(vec![2.0], Shape::new(vec![1])));
    let c = a.add(&b, session).await?;

    println!("✅ Compilation test passed!");

    return Ok(());
}
