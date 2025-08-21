use std::collections::VecDeque;
use std::error::Error;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot, watch};
use wgpu::{Adapter, AdapterInfo, BufferUsages, CommandBuffer, Device, DeviceType, Limits, Queue};

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

#[derive(Debug)]
pub struct ResourceLimits {
    pub max_concurrent_operations: usize,
    pub max_command_buffers: usize,
    pub max_memory_usage_mb: usize,
    pub operation_timeout: Duration,
    pub queue_depth_limit: usize,
}

impl ResourceLimits {
    fn estimate_safe_memory_mb(limits: &Limits) -> usize {
        let max_buffer_bytes = limits.max_buffer_size as usize;
        let safe_bytes = (max_buffer_bytes as f64 * 0.75) as usize;

        return safe_bytes / (1024 * 1024);
    }

    pub fn generic() -> Self {
        return Self {
            max_concurrent_operations: 8,
            max_command_buffers: 16,
            max_memory_usage_mb: 1024,
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 32,
        };
    }

    fn amd(limits: &Limits) -> Self {
        return Self {
            max_concurrent_operations: 4,
            max_command_buffers: 16,
            max_memory_usage_mb: Self::estimate_safe_memory_mb(limits),
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 16,
        };
    }

    fn amd_high(limits: &Limits) -> Self {
        return Self {
            max_concurrent_operations: 4, // HSA_MAX_QUEUES=4
            max_command_buffers: 32,
            max_memory_usage_mb: Self::estimate_safe_memory_mb(limits),
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 16,
        };
    }

    fn nvidia(limits: &Limits) -> Self {
        return Self {
            max_concurrent_operations: 12,
            max_command_buffers: 24,
            max_memory_usage_mb: Self::estimate_safe_memory_mb(limits),
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 48,
        };
    }

    pub fn autotune(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let limits = adapter.limits();

        match (info.device_type, info.vendor) {
            (DeviceType::DiscreteGpu, 0x1002) => {
                if info.name.contains("7900") {
                    return Self::amd_high(&limits);
                } else {
                    return Self::amd(&limits);
                }
            }
            (DeviceType::DiscreteGpu, 0x10DE) => {
                return Self::nvidia(&limits);
            }
            _ => {
                return Self::generic();
            }
        }
    }
}

#[derive(Debug)]
pub struct ResourceTracker {
    limits: ResourceLimits,
    current_operations: AtomicUsize,
    current_command_buffers: AtomicUsize,
    current_memory_usage_mb: AtomicUsize,
    queue_depth: AtomicUsize,
    emergency_stop: AtomicBool,
}

#[derive(Debug)]
pub struct ResourceUsage {
    pub gpu_operations: usize, // Operations currently executing on GPU
    pub command_buffers: usize,
    pub memory_usage_mb: usize,
    pub total_queue_depth: usize, // Total operations in system (executing + waiting)
}

impl ResourceTracker {
    pub fn new(limits: ResourceLimits) -> Result<Self, String> {
        return Ok(Self {
            limits,
            current_operations: AtomicUsize::new(0),
            current_command_buffers: AtomicUsize::new(0),
            current_memory_usage_mb: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            emergency_stop: AtomicBool::new(false),
        });
    }

    pub fn can_start_gpu_execution(&self) -> Result<(), GpuError> {
        if self.emergency_stop.load(Ordering::Acquire) {
            return Err(GpuError::CircuitBreakerOpen { failures: 0 });
        }

        let current_ops = self.current_operations.load(Ordering::Acquire);

        if current_ops >= self.limits.max_concurrent_operations {
            return Err(GpuError::QueueOverflow {
                active: current_ops,
                limit: self.limits.max_concurrent_operations,
            });
        }

        return Ok(());
    }

    pub fn can_queue_operation(&self) -> Result<(), GpuError> {
        if self.emergency_stop.load(Ordering::Acquire) {
            return Err(GpuError::CircuitBreakerOpen { failures: 0 });
        }

        let queue_depth = self.queue_depth.load(Ordering::Acquire);

        if queue_depth >= self.limits.queue_depth_limit {
            return Err(GpuError::QueueOverflow {
                active: queue_depth,
                limit: self.limits.queue_depth_limit,
            });
        }

        return Ok(());
    }

    pub fn can_allocate_command_buffer(&self) -> Result<(), GpuError> {
        let current = self.current_command_buffers.load(Ordering::Acquire);

        if current >= self.limits.max_command_buffers {
            return Err(GpuError::QueueOverflow {
                active: current,
                limit: self.limits.max_command_buffers,
            });
        }

        return Ok(());
    }

    pub fn can_allocate_memory(&self, size_mb: usize) -> Result<(), GpuError> {
        let current = self.current_memory_usage_mb.load(Ordering::Acquire);
        let new_total = current + size_mb;

        if new_total > self.limits.max_memory_usage_mb {
            return Err(GpuError::MemoryExhaustion {
                requested: size_mb,
                available: self.limits.max_memory_usage_mb.saturating_sub(current),
            });
        }

        return Ok(());
    }

    pub fn start_gpu_execution(
        &'_ self,
        _queue_guard: &QueueSlotGuard,
    ) -> Result<GpuExecutionGuard<'_>, GpuError> {
        self.can_start_gpu_execution()?;

        let prev_ops = self.current_operations.fetch_add(1, Ordering::AcqRel);

        if prev_ops >= self.limits.max_concurrent_operations {
            self.current_operations.fetch_sub(1, Ordering::AcqRel);

            return Err(GpuError::QueueOverflow {
                active: prev_ops + 1,
                limit: self.limits.max_concurrent_operations,
            });
        }

        return Ok(GpuExecutionGuard {
            tracker: self,
            _phantom: std::marker::PhantomData,
        });
    }

    pub fn reserve_queue_slot(&'_ self) -> Result<QueueSlotGuard<'_>, GpuError> {
        self.can_queue_operation()?;

        let prev_queue = self.queue_depth.fetch_add(1, Ordering::AcqRel);

        if prev_queue >= self.limits.queue_depth_limit {
            self.queue_depth.fetch_sub(1, Ordering::AcqRel);
            return Err(GpuError::QueueOverflow {
                active: prev_queue + 1,
                limit: self.limits.queue_depth_limit,
            });
        }

        return Ok(QueueSlotGuard {
            tracker: self,
            _phantom: std::marker::PhantomData,
        });
    }

    pub fn reserve_operation(&'_ self) -> Result<OperationGuard<'_>, GpuError> {
        let queue_guard = self.reserve_queue_slot()?;
        let gpu_guard = self.start_gpu_execution(&queue_guard)?;

        return Ok(OperationGuard {
            queue_guard,
            gpu_guard,
        });
    }

    pub fn reserve_command_buffer(&'_ self) -> Result<CommandBufferGuard<'_>, GpuError> {
        self.can_allocate_command_buffer()?;

        let prev = self.current_command_buffers.fetch_add(1, Ordering::AcqRel);

        if prev >= self.limits.max_command_buffers {
            self.current_command_buffers.fetch_sub(1, Ordering::AcqRel);

            return Err(GpuError::QueueOverflow {
                active: prev + 1,
                limit: self.limits.max_command_buffers,
            });
        }

        return Ok(CommandBufferGuard {
            tracker: self,
            _phantom: std::marker::PhantomData,
        });
    }

    pub fn reserve_memory(&'_ self, size_mb: usize) -> Result<MemoryGuard<'_>, GpuError> {
        self.can_allocate_memory(size_mb)?;

        self.current_memory_usage_mb
            .fetch_add(size_mb, Ordering::AcqRel);

        return Ok(MemoryGuard {
            tracker: self,
            size_mb,
            _phantom: std::marker::PhantomData,
        });
    }

    pub fn current_usage(&self) -> ResourceUsage {
        return ResourceUsage {
            gpu_operations: self.current_operations.load(Ordering::Acquire),
            command_buffers: self.current_command_buffers.load(Ordering::Acquire),
            memory_usage_mb: self.current_memory_usage_mb.load(Ordering::Acquire),
            total_queue_depth: self.queue_depth.load(Ordering::Acquire),
        };
    }

    pub fn emergency_stop_trigger(&self) {
        self.emergency_stop.store(true, Ordering::Release);
    }

    pub fn emergency_stop_reset(&self) {
        self.emergency_stop.store(false, Ordering::Release);
    }

    pub fn get_operation_timeout(&self) -> Duration {
        return self.limits.operation_timeout;
    }
}

/// RAII guard for queue slot reservation
pub struct QueueSlotGuard<'a> {
    tracker: &'a ResourceTracker,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Drop for QueueSlotGuard<'a> {
    fn drop(&mut self) {
        self.tracker.queue_depth.fetch_sub(1, Ordering::AcqRel);
    }
}

/// RAII guard for GPU execution
pub struct GpuExecutionGuard<'a> {
    tracker: &'a ResourceTracker,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Drop for GpuExecutionGuard<'a> {
    fn drop(&mut self) {
        self.tracker
            .current_operations
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// RAII guard for complete operation (queue + GPU execution)
pub struct OperationGuard<'a> {
    queue_guard: QueueSlotGuard<'a>,
    gpu_guard: GpuExecutionGuard<'a>,
}

impl<'a> OperationGuard<'a> {
    pub fn split(self) -> (QueueSlotGuard<'a>, GpuExecutionGuard<'a>) {
        let manual = ManuallyDrop::new(self);

        let queue_guard = unsafe { std::ptr::read(&manual.queue_guard) };
        let gpu_guard = unsafe { std::ptr::read(&manual.gpu_guard) };

        return (queue_guard, gpu_guard);
    }
}

/// RAII guard for command buffer
pub struct CommandBufferGuard<'a> {
    tracker: &'a ResourceTracker,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Drop for CommandBufferGuard<'a> {
    fn drop(&mut self) {
        self.tracker
            .current_command_buffers
            .fetch_sub(1, Ordering::AcqRel);
    }
}

/// RAII guard for memory
pub struct MemoryGuard<'a> {
    tracker: &'a ResourceTracker,
    size_mb: usize,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Drop for MemoryGuard<'a> {
    fn drop(&mut self) {
        self.tracker
            .current_memory_usage_mb
            .fetch_sub(self.size_mb, Ordering::AcqRel);
    }
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
