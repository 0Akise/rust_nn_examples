struct TensorElement {
    value: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<TensorElement>;
@group(0) @binding(1) var<storage, read> input_b: array<TensorElement>;
@group(0) @binding(2) var<storage, read_write> output: array<TensorElement>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = {{TOTAL_ELEMENTS}}u;
    
    if (index < total_elements) {
        output[index].value = input_a[index].value * input_b[index].value;
    }
}
