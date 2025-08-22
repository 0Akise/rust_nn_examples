use super::migration::Shape;

use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use dashmap::DashMap;
use wgpu::{BindGroupLayout, ComputePipeline};

//
// Shader
//

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConvolutionType {
    Forward,        // Standard convolution
    BackwardData,   // Gradient w.r.t. input
    BackwardFilter, // Gradient w.r.t. weights
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PaddingMode {
    Valid, // No padding
    Same,  // Output same size as input
    Explicit {
        // Manual padding specification
        pad_h_before: u32,
        pad_h_after: u32,
        pad_w_before: u32,
        pad_w_after: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataFormat {
    NCHW, // Batch, Channels, Height, Width (typical for GPU)
    NHWC, // Batch, Height, Width, Channels (typical for mobile)
    CHW,  // Channels, Height, Width (single batch)
    HWC,  // Height, Width, Channels (single batch)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvParams {
    pub conv_type: ConvolutionType, // Convolution type

    pub kernel_size: [u32; 2], // [height, width]
    pub stride: [u32; 2],      // [stride_h, stride_w]
    pub padding: PaddingMode,
    pub dilation: [u32; 2], // [dilation_h, dilation_w]

    pub input_channels: u32,
    pub output_channels: u32,
    pub groups: u32, // 1 = normal conv, input_channels = depthwise

    pub data_format: DataFormat, // Data layout

    pub use_winograd: bool,         // For 3x3 convolutions
    pub prefer_shared_memory: bool, // Cache input tiles in shared memory
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConvShaderVariant {
    Standard {
        kernel_size: [u32; 2],
        stride: [u32; 2],
        use_shared_memory: bool,
    },
    Pointwise, // Optimized 1x1 convolution
    Depthwise {
        kernel_size: [u32; 2],
    }, // Depthwise separable convolution
    Winograd3x3, // Winograd algorithm for 3x3
    Grouped {
        groups: u32,
    }, // Grouped convolution
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    ElementwiseAdd,
    ElementwiseMul,
    MatrixMultiply,
    Transpose,
    Reduce(ReduceOp),
    Convolution(ConvParams),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayoutKey {
    operation: OperationType,
    binding_count: usize,
    uniform_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    operation: OperationType,
    input_shapes: Vec<Shape>,
    workgroup_size: [u32; 3],
    device_id: u64, // Cache per device
}

#[derive(Debug, Clone)]
pub enum TemplateParam {
    WorkgroupSize {
        name: String,
        default: u32,
    },
    TotalElements {
        name: String,
    },
    MatrixDims {
        m: String,
        n: String,
        k: String,
    },
    InitValue {
        name: String,
        default: f32,
    },
    BufferCount {
        name: String,
    },
    ConvolutionKernel {
        kernel_h: String,
        kernel_w: String,
        values: [u32; 2],
    },
    ConvolutionStride {
        stride_h: String,
        stride_w: String,
        values: [u32; 2],
    },
    Channels {
        input_channels: String,
        output_channels: String,
        input_val: u32,
        output_val: u32,
    },
    Padding {
        name: String,
        value: u32,
    },
}

pub struct WorkgroupConstraints {
    max_x: u32,
    max_y: u32,
    max_z: u32,
    alignment_requirement: u32, // 64 for AMD wave alignment
    shared_memory_limit: u32,   // 32KB typically
}

pub struct ShaderTemplate {
    source: &'static str,           // Embedded WGSL source
    parameters: Vec<TemplateParam>, // What can be customized
    workgroup_constraints: WorkgroupConstraints,
    required_features: Vec<wgpu::Features>,
}

#[derive(thiserror::Error, Debug)]
pub enum ShaderError {
    #[error("Template not found for operation: {0:?}")]
    TemplateNotFound(OperationType),

    #[error("Workgroup size {actual:?} exceeds limit {limit:?}")]
    WorkgroupSizeExceeded { actual: [u32; 3], limit: [u32; 3] },

    #[error("Shared memory usage {used} exceeds limit {limit}")]
    SharedMemoryExceeded { used: u32, limit: u32 },

    #[error("WGSL compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Feature {0:?} not supported on this device")]
    UnsupportedFeature(wgpu::Features),
}

pub struct ShaderCompilationStats {
    total_compilations: AtomicU64,
    cache_hits: AtomicU64,
    total_compile_time_ms: AtomicU64,
    failed_compilations: AtomicU64,
}

pub struct ShaderManager {
    pipeline_cache: DashMap<PipelineKey, Arc<ComputePipeline>>, // Compiled pipeline cache
    shader_templates: HashMap<OperationType, ShaderTemplate>,   // Template source storage
    layout_cache: DashMap<LayoutKey, Arc<BindGroupLayout>>,     // Bind group layout cache
    compilation_stats: ShaderCompilationStats,                  // Compilation metrics
}

#[derive(Debug, Clone)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

#[derive(Debug, Clone)]
pub struct ShaderParams {
    pub operation: OperationType,
    pub input_shapes: Vec<Shape>,
    pub output_shape: Shape,
    pub workgroup_size: [u32; 3],
    pub precision: Precision,
}

pub mod shaders {
    pub const ADD: &str = include_str!("shaders/add.wgsl");
    pub const MUL: &str = include_str!("shaders/mul.wgsl");
    pub const MATMUL: &str = include_str!("shaders/matmul.wgsl");
    pub const TRANSPOSE: &str = include_str!("shaders/transpose.wgsl");
    pub const REDUCE_SUM: &str = include_str!("shaders/reduce_sum.wgsl");
}
