use std::collections::VecDeque;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8};
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot, watch};
use wgpu::{AdapterInfo, BufferUsages, CommandBuffer, Device, Queue};

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
