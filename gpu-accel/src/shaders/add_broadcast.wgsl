struct TensorElement {
    value: f32,
}

struct BroadcastInfo {
    shape_a: array<u32, 4>,
    shape_b: array<u32, 4>,
    shape_out: array<u32, 4>,
    rank_a: u32,
    rank_b: u32,
    rank_out: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<TensorElement>;
@group(0) @binding(1) var<storage, read> input_b: array<TensorElement>;
@group(0) @binding(2) var<storage, read_write> output: array<TensorElement>;
@group(0) @binding(3) var<uniform> broadcast_info: BroadcastInfo;

fn get_broadcasted_index(flat_idx: u32, shape: array<u32, 4>, rank: u32) -> u32 {
    var result = 0u;
    var remaining = flat_idx;
    var stride = 1u;
    
    for (var i = 0u; i < rank; i++) {
        let dim_idx = rank - 1u - i;
        let coord = remaining % shape[dim_idx];
        
        result += coord * stride;
        remaining = remaining / shape[dim_idx];
        stride *= shape[dim_idx];
    }
    
    return result;
}

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_elements = {{TOTAL_ELEMENTS}}u;
    
    if (index < total_elements) {
        let idx_a = get_broadcasted_index(index, broadcast_info.shape_a, broadcast_info.rank_a);
        let idx_b = get_broadcasted_index(index, broadcast_info.shape_b, broadcast_info.rank_b);
        
        output[index].value = input_a[idx_a].value + input_b[idx_b].value;
    }
}