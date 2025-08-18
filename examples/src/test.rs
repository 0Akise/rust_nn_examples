use nn_backbone::autograd::test;

fn main() {
    pollster::block_on(async {
        if let Err(e) = test::compilation().await {
            println!("Error: {}", e);
        }

        if let Err(e) = test::addition().await {
            println!("Error: {}", e);
        }

        if let Err(e) = test::matmul().await {
            println!("Error: {}", e);
        }
    });
}
