use super::Variable;

use std::sync::{Arc, Mutex};

use gpu_accel::{GpuSession, Shape, Tensor};

pub async fn compilation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing that everything compiles...");

    let session = Arc::new(Mutex::new(GpuSession::new().await.unwrap()));
    let a = Variable::with_grad(Tensor::new(vec![1.0], Shape::new(vec![1])));
    let b = Variable::with_grad(Tensor::new(vec![2.0], Shape::new(vec![1])));
    let _ = a.add(&b, session).await.unwrap();

    println!("✅ Compilation test passed!");

    return Ok(());
}

pub async fn operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Simple Addition Autograd");

    let session = Arc::new(Mutex::new(GpuSession::new().await.unwrap()));
    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0], Shape::new(vec![2])));
    let b = Variable::with_grad(Tensor::new(vec![3.0, 4.0], Shape::new(vec![2])));

    println!("a: {:?}", a.tensor.data);
    println!("b: {:?}", b.tensor.data);

    let c = a.add(&b, session.clone()).await.unwrap();

    println!("c = a + b: {:?}", c.tensor.data);
    println!("\n🔄 Starting backward pass...");

    let mut c_mut = c;

    c_mut.backward();

    println!("✅ Backward pass completed!");
    println!("Variable c has grad_fn: {}", c_mut.grad_fn.is_some());
    println!("\n🧪 Testing Matrix Multiplication Autograd");

    let a = Variable::with_grad(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![2, 2]),
    ));
    let b = Variable::with_grad(Tensor::new(
        vec![0.5, 1.0, 1.5, 2.0],
        Shape::new(vec![2, 2]),
    ));

    println!("Matrix A: {:?}", a.tensor.data);
    println!("Matrix B: {:?}", b.tensor.data);

    let c = a.matmul(&b, session.clone()).await.unwrap();

    println!("C = A @ B: {:?}", c.tensor.data);

    let mut c_mut = c;

    c_mut.backward();

    println!("✅ Matrix multiplication autograd completed!");
    println!("\n🧪 Testing Dot Product Autograd");

    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
    let b = Variable::with_grad(Tensor::new(vec![4.0, 5.0, 6.0], Shape::new(vec![3])));
    let c = a.dot(&b, session.clone()).await.unwrap();

    println!("C = A ‣ B: {:?}", c.tensor.data);
    println!("\n🔄 Starting backward pass...");

    let mut c_mut = c;

    c_mut.backward();

    println!("✅ Dot product autograd completed!");
    println!("\n🧪 Testing Transpose Autograd");

    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3])));
    let a_t = a.transpose(session.clone()).await.unwrap();

    println!(
        "1D vector: {:?}, shape: {:?}",
        a.tensor.data, a.tensor.shape.dims
    );
    println!(
        "1D vector: {:?}, shape: {:?}",
        a_t.tensor.data, a_t.tensor.shape.dims
    );

    let b = Variable::with_grad(Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0],
        Shape::new(vec![2, 2]),
    ));
    let b_t = b.transpose(session.clone()).await.unwrap();

    println!(
        "2D Matrix: {:?}, shape: {:?}",
        b.tensor.data, b.tensor.shape.dims
    );
    println!(
        "2D Matrix: {:?}, shape: {:?}",
        b_t.tensor.data, b_t.tensor.shape.dims
    );
    println!("\n🔄 Starting backward pass...");

    let mut b_t_mut = b_t;

    b_t_mut.backward();

    println!("✅ Transpose autograd completed!");

    return Ok(());
}
