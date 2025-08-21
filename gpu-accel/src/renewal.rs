use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8};
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot, watch};
use wgpu::{
    AdapterInfo, BindGroupLayout, BufferUsages, CommandBuffer, ComputePipeline, Device, Queue,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);
static BUFFER_COUNTER: AtomicU64 = AtomicU64::new(0);
static VARIABLE_COUNTER: AtomicU64 = AtomicU64::new(0);
static NODE_COUNTER: AtomicU64 = AtomicU64::new(0);

//
// Primitives
//

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub data: Arc<Vec<f32>>,
    pub shape: Shape,
}

//
// GPU
//

#[derive(Debug, Clone)]
pub enum GradientTask {
    // lightweight and contain only data needed for the specific operation
    Add {
        output_grad: TensorData,
        target_var: VariableId,
        operation_id: OperationId,
    },
    Mul {
        output_grad: TensorData,
        other_input: TensorData,
        target_var: VariableId,
        operation_id: OperationId,
    },
    MatMul {
        output_grad: TensorData,
        other_input: TensorData,
        transpose_other: bool,
        target_var: VariableId,
        operation_id: OperationId,
    },
    Transpose {
        output_grad: TensorData,
        target_var: VariableId,
        operation_id: OperationId,
    },
}

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Success,
    Failed(String),
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    Direct,    // Allocate immediately when needed
    Pooled,    // Use pre-allocated buffer pool
    Lazy,      // Allocate on first use
    Streaming, // For large sequential operations
}

#[derive(Debug, Clone)]
pub enum AccumulationStrategy {
    Sum,                          // Default: just add gradients
    Average,                      // Divide by accumulation count
    WeightedAverage,              // Time-decay weighted average
    ClippedSum { max_norm: f32 }, // Gradient clipping
}

#[derive(Debug)]
pub enum ComputationResult {
    Tensor(TensorData),
    Scalar(f32),
    Error(String),
    Pending,
}

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("Operation timeout after {timeout:?}")]
    Timeout {
        timeout: Duration,
        operation_id: OperationId,
    },

    #[error("Queue overflow: {active}/{limit} operations")]
    QueueOverflow { active: usize, limit: usize },

    #[error("Memory exhaustion: {requested}MB, available: {available}MB")]
    MemoryExhaustion { requested: usize, available: usize },

    #[error("Circuit breaker open: {failures} consecutive failures")]
    CircuitBreakerOpen { failures: usize },
}

pub struct ResourceLimits {
    pub max_concurrent_operations: usize,
    pub max_command_buffers: usize,
    pub max_memory_usage_mb: usize,
    pub operation_timeout: Duration,
    pub queue_depth_limit: usize,
}

pub struct ActiveCommandBuffer {
    buffer: CommandBuffer,
    created_at: Instant,
    operation_id: OperationId,
    timeout: Duration,
}

pub struct CommandBufferPool {
    available_encoders: VecDeque<wgpu::CommandEncoder>,
    active_buffers: Vec<ActiveCommandBuffer>,
    batch_size: usize,
    submission_counter: AtomicUsize,
}

#[derive(Debug)]
pub struct TaskCompletion {
    pub operation_id: OperationId,
    pub variable_id: VariableId,
    pub status: TaskStatus,
    pub duration: Duration,
    pub gradient_norm: Option<f32>,
}

pub struct GradientTaskManager {
    task_queue: flume::Receiver<GradientTask>,
    task_sender: flume::Sender<GradientTask>,
    active_tasks: Arc<AtomicUsize>,
    max_concurrent_tasks: usize,
    completion_notifier: watch::Sender<TaskCompletion>,
    circuit_breaker: CircuitBreaker,
}

pub struct CircuitBreaker {
    failure_count: AtomicUsize,
    failure_threshold: usize,
    recovery_timeout: Duration,
    last_failure: AtomicU64,
    state: AtomicU8,
}

pub struct BufferInfo {
    size: usize,
    usage: BufferUsages,
    created_at: Instant,
    last_used: AtomicU64,
    ref_count: AtomicUsize,
}

pub struct GpuMemoryManager {
    allocated_buffers: DashMap<BufferId, BufferInfo>,
    memory_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    allocation_strategy: AllocationStrategy,
    cleanup_scheduler: mpsc::Sender<BufferId>,
}

#[derive(Debug, Clone)]
pub struct CancellationToken {
    pub is_cancelled: Arc<AtomicBool>,
    pub reason: Arc<std::sync::Mutex<Option<String>>>,
}

pub struct OperationTimeout {
    operation_id: OperationId,
    timeout: Duration,
    started_at: Instant,
    cancellation_token: CancellationToken,
}

pub struct GpuMetrics {
    operations_completed: AtomicU64,
    operations_failed: AtomicU64,
    average_latency_ms: AtomicU64,
    memory_usage_mb: AtomicUsize,
    queue_depth: AtomicUsize,
    last_heartbeat: AtomicU64,
}

#[derive(Debug)]
pub struct QueueDepthSample {
    pub timestamp: Instant,
    pub depth: usize,
    pub pending_operations: usize,
    pub memory_usage_mb: usize,
}

#[derive(Debug)]
pub struct PerformanceCounter {
    pub total_operations: AtomicU64,
    pub successful_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub total_compute_time_ms: AtomicU64,
    pub peak_memory_usage_mb: AtomicUsize,
    pub average_queue_depth: AtomicUsize,
}

pub struct GpuHealthMonitor {
    metrics: GpuMetrics,
    queue_depth_tracker: VecDeque<QueueDepthSample>,
    performance_counter: PerformanceCounter,
}

pub struct GpuContext {
    pub adapter_info: AdapterInfo,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub command_pool: CommandBufferPool,
    pub memory_manager: GpuMemoryManager,
    pub resource_limits: ResourceLimits,
    pub health_monitor: GpuHealthMonitor,
}

#[derive(Debug)]
pub struct GradientEntry {
    pub gradient: TensorData,
    pub accumulation_count: AtomicUsize,
    pub last_updated: AtomicU64, // Unix timestamp
    pub overflow_detected: AtomicBool,
}

pub struct GradientAccumulator {
    gradients: DashMap<VariableId, GradientEntry>,
    accumulation_strategy: AccumulationStrategy,
    overflow_protection: bool,
}

pub struct Node {
    dependencies_remaining: AtomicUsize,
    computation_status: AtomicU8,
    result: Option<ComputationResult>,
    waiters: Vec<oneshot::Sender<()>>,
}

#[derive(Debug)]
pub struct DependencyTracker {
    pub forward_deps: DashMap<NodeId, Vec<NodeId>>, // node -> its dependencies
    pub reverse_deps: DashMap<NodeId, Vec<NodeId>>, // node -> nodes depending on it
    pub completion_count: DashMap<NodeId, AtomicUsize>,
}

pub struct Graph {
    node_states: DashMap<NodeId, Node>,
    completion_order: Vec<NodeId>,
    dependency_tracker: DependencyTracker,
    gradient_accumulator: GradientAccumulator,
}

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
