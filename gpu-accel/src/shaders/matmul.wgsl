struct TensorElement {
    value: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<TensorElement>;
@group(0) @binding(1) var<storage, read> input_b: array<TensorElement>;
@group(0) @binding(2) var<storage, read_write> output: array<TensorElement>;

@compute @workgroup_size({{WORKGROUP_SIZE_X}}, {{WORKGROUP_SIZE_Y}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    let M = {{M}}u;
    let N = {{N}}u;
    let P = {{P}}u;
    
    if (row < M && col < P) {
        var sum = 0.0;

        for (var k = 0u; k < N; k++) {
            let a_idx = row * N + k;
            let b_idx = k * P + col;

            sum += input_a[a_idx].value * input_b[b_idx].value;
        }
        
        let output_idx = row * P + col;
        
        output[output_idx].value = sum;
    }
}