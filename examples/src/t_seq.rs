use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::gpu_context::GpuContext;
use nn_backbone::autograd::variable::Variable;
use nn_backbone::autograd::{COMPUTATION_GRAPH, GRADIENT_REGISTRY};
use nn_backbone::layer::loss::CrossEntropyLoss;
use nn_backbone::layer::{layers, model::Sequential};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn test_sequential() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Sequential Model Graph Building\n");

    // Clear state
    COMPUTATION_GRAPH.clear().await;
    GRADIENT_REGISTRY.clear_gradients().await;

    let context = Arc::new(Mutex::new(GpuContext::new().await?));

    // Build tiny model
    println!("Building model...");
    let mut model = Sequential::new()
        .add(layers::linear(4, 3, context.clone()).await?)
        .add(layers::relu())
        .add(layers::linear(3, 2, context.clone()).await?)
        .add(layers::softmax());

    println!("Model has {} layers", model.len());
    println!("Model has {} parameters", model.parameters().len());

    // Create tiny input
    let input = Variable::with_grad(Tensor::new(
        vec![
            0.1, 0.2, 0.3, 0.4, // Sample 1
            0.5, 0.6, 0.7, 0.8,
        ], // Sample 2
        Shape::new(vec![2, 4]),
    ));

    let targets = Variable::with_grad(Tensor::new(
        vec![
            1.0, 0.0, // Sample 1: class 0
            0.0, 1.0,
        ], // Sample 2: class 1
        Shape::new(vec![2, 2]),
    ));

    println!("\n📍 Forward pass through Sequential model...");
    let output = model.forward(&input).await?;

    println!("  Input shape: {:?}", input.shape());
    println!("  Output shape: {:?}", output.shape());
    println!("  Output ID: {}", output.id);
    println!("  Output has grad_fn: {}", output.grad_fn.is_some());
    println!("  Output parents: {:?}", output.parents);

    // Compute loss
    println!("\n📍 Computing CrossEntropyLoss...");
    let loss_fn = CrossEntropyLoss::new();
    let mut loss = loss_fn.forward(&output, &targets).await?;

    println!("  Loss value: {}", loss.tensor.data[0]);
    println!("  Loss ID: {}", loss.id);
    println!("  Loss has grad_fn: {}", loss.grad_fn.is_some());
    println!("  Loss parents: {:?}", loss.parents);

    // Check computation graph
    println!("\n📊 Analyzing computation graph...");
    let backward_order = COMPUTATION_GRAPH.get_backward_order(loss.id).await;

    println!("  Backward order: {:?}", backward_order);
    println!("  Total nodes: {}", backward_order.len());

    if backward_order.len() > 1 {
        println!("  ✅ Graph is properly connected!");

        // Analyze each node
        println!("\n  Node details:");
        for id in &backward_order {
            if let Some(node) = COMPUTATION_GRAPH.get_node(*id).await {
                println!(
                    "    Node {}: has_grad_fn={}, parents={:?}",
                    id,
                    node.grad_fn.is_some(),
                    node.parents
                );
            }
        }

        // Try backward with smaller timeout
        println!("\n📍 Starting backward pass...");
        let backward_result = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
            let ctx = context.lock().await;
            ctx.backward(&mut loss).await
        })
        .await;

        match backward_result {
            Ok(Ok(())) => {
                println!("  ✅ Backward completed!");

                // Check gradients
                println!("\n📊 Checking parameter gradients:");
                for (i, param) in model.parameters().iter().enumerate() {
                    if let Some(grad) = GRADIENT_REGISTRY.get_gradient(param.id).await {
                        let grad_norm = grad.data.iter().map(|x| x * x).sum::<f32>().sqrt();
                        println!(
                            "    Param {} (var {}): grad norm = {:.6}",
                            i, param.id, grad_norm
                        );
                    } else {
                        println!("    Param {} (var {}): NO GRADIENT", i, param.id);
                    }
                }
            }
            Ok(Err(e)) => {
                println!("  ❌ Backward failed: {}", e);
            }
            Err(_) => {
                println!("  ⏱️ Backward timed out!");
                println!("  The hang is in the backward pass.");
            }
        }
    } else {
        println!("  ❌ Graph not properly connected!");
        println!("  Only {} node(s) found", backward_order.len());
        println!("  This means the loss is not connected to the model parameters.");
    }

    println!("\n🎯 Test complete!");
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = test_sequential().await {
        eprintln!("❌ Test failed: {}", e);
    }
}
