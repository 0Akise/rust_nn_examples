struct TensorElement {
    value: f32,
}

@group(0) @binding(0) var<storage, read> input: array<TensorElement>;
@group(0) @binding(1) var<storage, read_write> output: array<TensorElement>;

@compute @workgroup_size({{WORKGROUP_SIZE}}, {{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    let ROWS = {{ROWS}}u;
    let COLS = {{COLS}}u;
    
    if (row < ROWS && col < COLS) {
        let input_idx = row * COLS + col;
        let output_idx = col * ROWS + row;
        output[output_idx].value = input[input_idx].value;
    }
}