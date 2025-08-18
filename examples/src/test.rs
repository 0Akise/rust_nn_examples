use nn_backbone::autograd::test;

fn main() {
    pollster::block_on(async {
        if let Err(e) = test::compilation().await {
            println!("Error: {}", e);
        }

        if let Err(e) = test::operations().await {
            println!("Error: {}", e);
        }
    });
}
