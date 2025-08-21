use gpu_accel::{Shape, Tensor};
use nn_backbone::autograd::gpu_context::GpuContext;
use nn_backbone::autograd::variable::Variable;
use nn_backbone::autograd::{COMPUTATION_GRAPH, GRADIENT_REGISTRY};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn test_simple() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Simple Gradient Computation\n");

    // Clear any previous state
    COMPUTATION_GRAPH.clear().await;
    GRADIENT_REGISTRY.clear_gradients().await;

    let context = Arc::new(Mutex::new(GpuContext::new().await?));

    // Test 1: Simple addition
    println!("Test 1: Simple addition");
    let a = Variable::with_grad(Tensor::new(vec![1.0, 2.0], Shape::new(vec![2])));
    let b = Variable::with_grad(Tensor::new(vec![3.0, 4.0], Shape::new(vec![2])));

    // Register variables in graph
    a.register_in_graph().await;
    b.register_in_graph().await;

    let c = {
        let ctx = context.lock().await;
        ctx.forward_add(&a, &b).await?
    };

    println!("  a (id {}): [1, 2]", a.id);
    println!("  b (id {}): [3, 4]", b.id);
    println!("  c = a + b (id {}): {:?}", c.id, c.tensor.data);
    println!("  c has grad_fn: {}", c.grad_fn.is_some());
    println!("  c parents: {:?}", c.parents);

    // Create a simple loss (sum of c)
    let loss_value = c.tensor.data.iter().sum::<f32>();
    let mut loss = Variable::with_grad(Tensor::new(vec![1.0, 1.0], Shape::new(vec![2])));
    loss.parents = vec![c.id];
    loss.register_in_graph().await;

    // Check graph
    let order = COMPUTATION_GRAPH.get_backward_order(loss.id).await;
    println!("\n  Graph order from loss: {:?}", order);

    if order.len() > 1 {
        println!("  ✅ Graph properly connected!");

        // Try backward
        println!("\n  Starting backward...");
        {
            let ctx = context.lock().await;
            ctx.backward(&mut loss).await?;
        }

        // Check gradients
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        if let Some(grad_a) = GRADIENT_REGISTRY.get_gradient(a.id).await {
            println!("  Gradient of a: {:?}", grad_a.data);
        } else {
            println!("  No gradient for a");
        }

        if let Some(grad_b) = GRADIENT_REGISTRY.get_gradient(b.id).await {
            println!("  Gradient of b: {:?}", grad_b.data);
        } else {
            println!("  No gradient for b");
        }
    } else {
        println!("  ❌ Graph not connected properly!");
    }

    // Test 2: Simple matmul
    println!("\n\nTest 2: Simple matmul");
    COMPUTATION_GRAPH.clear().await;
    GRADIENT_REGISTRY.clear_gradients().await;

    let x = Variable::with_grad(Tensor::new(vec![1.0, 2.0], Shape::new(vec![1, 2])));
    let w = Variable::with_grad(Tensor::new(vec![3.0, 4.0], Shape::new(vec![2, 1])));

    x.register_in_graph().await;
    w.register_in_graph().await;

    let y = {
        let ctx = context.lock().await;
        ctx.forward_matmul(&x, &w).await?
    };

    println!("  x (id {}): [[1, 2]]", x.id);
    println!("  w (id {}): [[3], [4]]", w.id);
    println!("  y = x @ w (id {}): {:?}", y.id, y.tensor.data);
    println!("  y has grad_fn: {}", y.grad_fn.is_some());
    println!("  y parents: {:?}", y.parents);

    // Use y directly as loss
    let mut loss = y;

    let order = COMPUTATION_GRAPH.get_backward_order(loss.id).await;
    println!("\n  Graph order from loss: {:?}", order);

    if order.len() >= 3 {
        // Should have loss, x, and w
        println!("  ✅ MatMul graph properly connected!");

        println!("\n  Starting backward...");
        {
            let ctx = context.lock().await;
            ctx.backward(&mut loss).await?;
        }

        // Wait and check gradients
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;

        if let Some(grad_x) = GRADIENT_REGISTRY.get_gradient(x.id).await {
            println!("  Gradient of x: {:?}", grad_x.data);
        } else {
            println!("  No gradient for x");
        }

        if let Some(grad_w) = GRADIENT_REGISTRY.get_gradient(w.id).await {
            println!("  Gradient of w: {:?}", grad_w.data);
        } else {
            println!("  No gradient for w");
        }
    } else {
        println!("  ❌ MatMul graph not connected properly!");
    }

    println!("\n🎯 Test complete!");
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = test_simple().await {
        eprintln!("❌ Test failed: {}", e);
    }
}
