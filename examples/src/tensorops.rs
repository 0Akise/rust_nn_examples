use gpu_accel::{GpuSession, Shape, Tensor};

fn print_matrix(name: &str, tensor: &Tensor) {
    println!("\n{}: {:?}", name, tensor.shape.dims);

    if tensor.shape.rank() == 2 {
        let rows = tensor.shape.dims[0];
        let cols = tensor.shape.dims[1];

        for row in 0..rows {
            print!("  [");

            for col in 0..cols {
                let idx = row * cols + col;

                if col == cols - 1 {
                    print!("{:6.2}", tensor.data[idx]);
                } else {
                    print!("{:6.2}, ", tensor.data[idx]);
                }
            }
            println!("]");
        }
    } else {
        println!("  Data: {:?}", tensor.data);
    }
}

async fn tensorops_demo() {
    let mut gpu_session = GpuSession::new()
        .await
        .expect("Failed to create GPU session");

    let matrix_a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    let matrix_b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![3, 2]));
    let matrix_c = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2]));

    print_matrix("Matrix A (2x3)", &matrix_a);
    print_matrix("Matrix B (3x2)", &matrix_b);
    print_matrix("Matrix C (2x2)", &matrix_c);

    let result1 = gpu_session.matmul(&matrix_a, &matrix_b).await.unwrap();
    print_matrix("A x B (2x2)", &result1);

    let result2 = gpu_session.add(&result1, &matrix_c).await.unwrap();
    print_matrix("(A x B) + C", &result2);

    let result3 = gpu_session.multiply(&result1, &matrix_c).await.unwrap();
    print_matrix("(A x B) ⊙ C", &result3);

    let transpose_a = gpu_session.transpose(&matrix_a).await.unwrap();
    print_matrix("A^T (3x2)", &transpose_a);

    let transpose_b = gpu_session.transpose(&matrix_b).await.unwrap();
    print_matrix("B^T (2x3)", &transpose_b);
}

fn main() {
    pollster::block_on(tensorops_demo());
}
