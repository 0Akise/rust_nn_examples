struct TensorElement {
    value: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<TensorElement>;
@group(0) @binding(1) var<storage, read> input_b: array<TensorElement>;
@group(0) @binding(2) var<storage, read_write> output: array<TensorElement>;

var<workgroup> shared_data: array<f32, {{WORKGROUP_SIZE}}>;

@compute @workgroup_size({{WORKGROUP_SIZE}})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let total_elements = {{TOTAL_ELEMENTS}}u;
    let tid = local_id.x;
    let gid = global_id.x;
    let workgroup_size = {{WORKGROUP_SIZE}}u;
    
    if (gid < total_elements) {
        shared_data[tid] = input_a[gid].value * input_b[gid].value;
    } else {
        shared_data[tid] = 0.0;
    }
    
    workgroupBarrier();
    
    var stride = workgroup_size / 2u;

    while (stride > 0u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
    
        workgroupBarrier();
    
        stride = stride / 2u;
    }
    
    if (tid == 0u) {
        output[group_id.x].value = shared_data[0];
    }
}