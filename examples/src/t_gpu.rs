// In examples/src/test_backward.rs
// Fixed version with proper loss computation

use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::gpu_context::GpuContext;
use nn_backbone::autograd::variable::Variable;
use nn_backbone::layer::layers;
use nn_backbone::layer::loss::CrossEntropyLoss;
use nn_backbone::layer::model::Sequential;
use std::sync::Arc;
use tokio::sync::Mutex;

async fn test_backward() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Synchronous Backward Pass\n");

    // Create GPU context
    let context = Arc::new(Mutex::new(GpuContext::new().await?));

    println!("Testing backward pass with proper loss connection...\n");

    // Create a simple network
    let mut model = Sequential::new()
        .add(layers::linear(10, 5, context.clone()).await?)
        .add(layers::relu())
        .add(layers::linear(5, 2, context.clone()).await?)
        .add(layers::softmax()); // Add softmax for proper loss computation

    // Create small input
    let input = Variable::with_grad(Tensor::new(
        vec![0.5; 20], // 2 samples, 10 features each
        Shape::new(vec![2, 10]),
    ));

    // Create target labels (one-hot encoded)
    let targets = Variable::with_grad(Tensor::new(
        vec![
            1.0, 0.0, // Sample 1: class 0
            0.0, 1.0,
        ], // Sample 2: class 1
        Shape::new(vec![2, 2]),
    ));

    println!("📍 Forward pass...");
    let output = model.forward(&input).await?;
    println!("  Output shape: {:?}", output.shape());
    println!("  Output has grad_fn: {}", output.grad_fn.is_some());
    println!("  Output ID: {}, Parents: {:?}", output.id, output.parents);

    // Use CrossEntropyLoss to properly connect the loss to the computation graph
    println!("\n📍 Computing loss...");
    let loss_fn = CrossEntropyLoss::new();
    let mut loss = loss_fn.forward(&output, &targets).await?;
    println!("  Loss value: {}", loss.tensor.data[0]);
    println!("  Loss has grad_fn: {}", loss.grad_fn.is_some());
    println!("  Loss ID: {}, Parents: {:?}", loss.id, loss.parents);

    // Register loss in graph
    loss.register_in_graph().await;

    println!("\n📍 Backward pass (synchronous)...");
    {
        let ctx = context.lock().await;
        ctx.backward(&mut loss).await?;
    }

    println!("\n✅ Backward pass completed successfully!");

    // Check gradients
    println!("\n📊 Checking gradients:");
    for (i, param) in model.parameters().iter().enumerate() {
        if let Some(grad) = nn_backbone::autograd::GRADIENT_REGISTRY
            .get_gradient(param.id)
            .await
        {
            let grad_norm = grad.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "  Parameter {} (var {}): gradient norm = {:.6}",
                i, param.id, grad_norm
            );
        } else {
            println!("  Parameter {} (var {}): NO GRADIENT", i, param.id);
        }
    }

    println!("\n🎯 Test complete!");

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = test_backward().await {
        eprintln!("❌ Test failed: {}", e);
    }
}
