use gpu_accel::{GpuSession, Shape, Tensor};

async fn tensorops_demo() {
    let mut gpu_session = GpuSession::new()
        .await
        .expect("Failed to create GPU session");

    let matrix_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    let matrix_b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![3, 2]));
    let matrix_c = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2]));

    println!("Matrix A (2x3): {:?}", matrix_a.data);
    println!("Matrix B (3x2): {:?}", matrix_b.data);
    println!("Matrix C (2x2): {:?}", matrix_c.data);

    let result1 = gpu_session.matmul(&matrix_a, &matrix_b).await.unwrap();
    println!("Matrix multiply result: {:?}", result1.data);

    let result2 = gpu_session.add(&result1, &matrix_c).await.unwrap();
    println!("Add result: {:?}", result2.data);

    let result3 = gpu_session.multiply(&result1, &matrix_c).await.unwrap();
    println!("Element-wise multiply result: {:?}", result3.data);
}

fn main() {
    pollster::block_on(tensorops_demo());
}
