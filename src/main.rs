pub mod prelude;

use prelude::gpu_module::GpuModule;
use prelude::{Operation, Shape, Tensor};

async fn demonstrate_tensor_operations() {
    let gpu_module = GpuModule::new().await;

    let matrix_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    let matrix_b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![3, 2]));

    println!("Matrix A (2x3): {:?}", matrix_a.data);
    println!("Matrix B (3x2): {:?}", matrix_b.data);

    let result = gpu_module
        .unwrap()
        .binary_op(&matrix_a, &matrix_b, Operation::MatrixMultiply)
        .await
        .unwrap();

    println!("Result (2x2): {:?}", result.data);
    println!("Expected: [22, 28, 49, 64]");
}

fn main() {
    env_logger::init();
    pollster::block_on(demonstrate_tensor_operations());
}
