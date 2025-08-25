use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};
use wgpu::{Adapter, AdapterInfo, Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, DeviceType, Limits, Queue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);
static BUFFER_COUNTER: AtomicU64 = AtomicU64::new(0);
static VARIABLE_COUNTER: AtomicU64 = AtomicU64::new(0);
static NODE_COUNTER: AtomicU64 = AtomicU64::new(0);

//
// Helper functions
//

pub fn next_buffer_id() -> BufferId {
    BufferId(BUFFER_COUNTER.fetch_add(1, Ordering::Relaxed))
}

//
// Primitives
//

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape: Shape,
}

//
// GPU
//

#[derive(Debug, Clone)]
pub enum AccumulationStrategy {
    Sum,                          // Default: just add gradients
    Average,                      // Divide by accumulation count
    WeightedAverage,              // Time-decay weighted average
    ClippedSum { max_norm: f32 }, // Gradient clipping
}

#[derive(Debug)]
pub enum ComputationResult {
    Tensor(Tensor),
    Scalar(f32),
    Error(String),
    Pending,
}

#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("Operation timeout after {timeout:?}")]
    Timeout { timeout: Duration, operation_id: OperationId },

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
    pub fn estimate_memory(limits: &Limits) -> usize {
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
            max_memory_usage_mb: Self::estimate_memory(limits),
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 16,
        };
    }

    fn amd_high(limits: &Limits) -> Self {
        return Self {
            max_concurrent_operations: 4, // HSA_MAX_QUEUES=4
            max_command_buffers: 32,
            max_memory_usage_mb: Self::estimate_memory(limits),
            operation_timeout: Duration::from_secs(10),
            queue_depth_limit: 16,
        };
    }

    fn nvidia(limits: &Limits) -> Self {
        return Self {
            max_concurrent_operations: 12,
            max_command_buffers: 24,
            max_memory_usage_mb: Self::estimate_memory(limits),
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

    pub fn start_gpu_execution(&'_ self, _queue_guard: &QueueSlotGuard) -> Result<GpuExecutionGuard<'_>, GpuError> {
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

        return Ok(OperationGuard { queue_guard, gpu_guard });
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
        self.current_memory_usage_mb.fetch_add(size_mb, Ordering::AcqRel);

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
        self.tracker.current_operations.fetch_sub(1, Ordering::AcqRel);
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
        self.tracker.current_command_buffers.fetch_sub(1, Ordering::AcqRel);
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
        self.tracker.current_memory_usage_mb.fetch_sub(self.size_mb, Ordering::AcqRel);
    }
}

pub struct ActiveCommandBuffer {
    buffer: CommandBuffer,
    created_at: Instant,
    operation_id: OperationId,
    timeout: Duration,
}

impl ActiveCommandBuffer {
    pub fn new(buffer: CommandBuffer, operation_id: OperationId, timeout: Duration) -> Self {
        return Self {
            buffer,
            created_at: Instant::now(),
            operation_id,
            timeout,
        };
    }

    pub fn is_timed_out(&self) -> bool {
        return self.created_at.elapsed() > self.timeout;
    }

    pub fn get_remaining_time(&self) -> Duration {
        return self.timeout.saturating_sub(self.created_at.elapsed());
    }
}

#[derive(Debug)]
pub struct CommandBufferStatus {
    pub active_buffers: usize,
    pub available_encoders: usize,
    pub total_submissions: usize,
    pub timeout_count: usize,
    pub oldest_buffer_age: Option<Duration>,
}

pub struct CommandBufferPool {
    device: Device,
    queue: Queue,

    available_encoders: VecDeque<CommandEncoder>,
    active_buffers: Vec<ActiveCommandBuffer>,

    batch_size: usize,
    max_active_buffers: usize,
    default_timeout: Duration,

    pub submission_counter: AtomicUsize,
    pub total_submissions: AtomicUsize,
    pub timeout_count: AtomicUsize,
}

impl CommandBufferPool {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, batch_size: usize, max_active_buffers: usize, default_timeout: Duration) -> Self {
        return Self {
            device,
            queue,
            available_encoders: VecDeque::with_capacity(16),
            active_buffers: Vec::with_capacity(max_active_buffers),
            batch_size,
            max_active_buffers,
            default_timeout,
            submission_counter: AtomicUsize::new(0),
            total_submissions: AtomicUsize::new(0),
            timeout_count: AtomicUsize::new(0),
        };
    }

    /// Get a command encoder, either from pool or create new
    pub fn get_encoder(&mut self, label: Option<&str>, _resource_tracker: &ResourceTracker) -> Result<CommandEncoder, GpuError> {
        if let Some(encoder) = self.available_encoders.pop_front() {
            return Ok(encoder);
        } else {
            let encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label });
            return Ok(encoder);
        }
    }

    /// Cleanup timed out command buffers
    fn cleanup_timed_out_buffers(&mut self) {
        let initial_count = self.active_buffers.len();

        self.active_buffers.retain(|buffer| {
            if buffer.is_timed_out() {
                tracing::warn!(
                    operation_id = ?buffer.operation_id,
                    age = ?buffer.created_at.elapsed(),
                    timeout = ?buffer.timeout,
                    "Command buffer timed out and dropped"
                );

                self.timeout_count.fetch_add(1, Ordering::Relaxed);

                return false;
            } else {
                return true;
            }
        });

        let cleaned_count = initial_count - self.active_buffers.len();

        if 0 < cleaned_count {
            tracing::debug!(cleaned_buffers = cleaned_count, remaining_buffers = self.active_buffers.len(), "Cleaned up timed out command buffers");
        }
    }

    /// Check if we should submit the current batch
    fn should_submit_batch(&self) -> bool {
        if self.active_buffers.is_empty() == true {
            return false;
        }

        if self.batch_size <= self.active_buffers.len() {
            return true;
        }

        if let Some(oldest) = self.active_buffers.first() {
            let remaining = oldest.get_remaining_time();

            if remaining < Duration::from_millis(100) {
                return true;
            }
        }

        return false;
    }

    /// Force submission of all pending buffers
    pub fn submit_batch(&mut self) -> Result<usize, GpuError> {
        if self.active_buffers.is_empty() == true {
            return Ok(0);
        }

        let pre_cleanup_count = self.active_buffers.len();

        self.cleanup_timed_out_buffers();

        let post_cleanup_count = self.active_buffers.len();

        if self.active_buffers.is_empty() == true {
            return Ok(0);
        }

        let buffers: Vec<CommandBuffer> = self.active_buffers.drain(..).map(|active| active.buffer).collect();
        let count = buffers.len();

        self.queue.submit(buffers);
        self.submission_counter.fetch_add(1, Ordering::Relaxed);
        self.total_submissions.fetch_add(count, Ordering::Relaxed);

        tracing::info!(
            batch_size = count,
            timed_out = pre_cleanup_count - post_cleanup_count,
            submission_id = self.submission_counter.load(Ordering::Relaxed),
            "Command buffer batch submitted"
        );

        Ok(count)
    }

    /// Submit a command buffer to the pool for batched execution
    pub fn submit_buffer(&mut self, buffer: CommandBuffer, operation_id: OperationId, timeout: Option<Duration>, _guard: CommandBufferGuard<'_>) -> Result<(), GpuError> {
        if self.max_active_buffers <= self.active_buffers.len() {
            return Err(GpuError::QueueOverflow {
                active: self.active_buffers.len(),
                limit: self.max_active_buffers,
            });
        }

        let timeout = timeout.unwrap_or(self.default_timeout);
        let active_buffer = ActiveCommandBuffer::new(buffer, operation_id, timeout);

        self.active_buffers.push(active_buffer);

        if self.should_submit_batch() == true {
            self.submit_batch()?;
        }

        return Ok(());
    }

    /// Flush all pending operations immediately
    pub fn flush(&mut self) -> Result<usize, GpuError> {
        self.submit_batch()
    }

    /// Return encoder to pool for reuse (called when operation completes)
    pub fn return_encoder(&mut self, encoder: CommandEncoder) {
        if self.available_encoders.len() < 8 {
            self.available_encoders.push_back(encoder);
        }
    }

    pub fn get_pending_operations(&self) -> Vec<(OperationId, Duration, Duration)> {
        return self.active_buffers.iter().map(|buffer| (buffer.operation_id, buffer.created_at.elapsed(), buffer.get_remaining_time())).collect();
    }

    /// Emergency cleanup - drop all pending buffers
    pub fn emergency_cleanup(&mut self) -> usize {
        let count = self.active_buffers.len();

        if 0 < count {
            tracing::error!(dropped_buffers = count, "Emergency cleanup - dropping all pending command buffers");

            self.active_buffers.clear();
            self.available_encoders.clear();
        }

        count
    }

    /// Get current pool statistics
    pub fn get_status(&self) -> CommandBufferStatus {
        let oldest_buffer_age = self.active_buffers.first().map(|buffer| buffer.created_at.elapsed());

        CommandBufferStatus {
            active_buffers: self.active_buffers.len(),
            available_encoders: self.available_encoders.len(),
            total_submissions: self.total_submissions.load(Ordering::Relaxed),
            timeout_count: self.timeout_count.load(Ordering::Relaxed),
            oldest_buffer_age,
        }
    }

    /// Health check - returns true if pool is operating normally
    pub fn health_check(&self) -> bool {
        let stats = self.get_status();
        let timeout_rate;

        if 0 < stats.total_submissions {
            timeout_rate = (stats.timeout_count as f64) / (stats.total_submissions as f64)
        } else {
            timeout_rate = 0.0
        };

        let healthy = timeout_rate < 0.1 && stats.active_buffers < self.max_active_buffers;

        if healthy == false {
            tracing::warn!(
                timeout_rate = timeout_rate,
                active_buffers = stats.active_buffers,
                max_buffers = self.max_active_buffers,
                "Command buffer pool health check failed"
            );
        }

        return healthy;
    }
}

// RAII for automatic encoder management
pub struct ManagedEncoder<'a> {
    encoder: Option<CommandEncoder>,
    pool: &'a mut CommandBufferPool,
    operation_id: OperationId,
}

impl<'a> ManagedEncoder<'a> {
    pub fn new(pool: &'a mut CommandBufferPool, operation_id: OperationId, label: Option<&str>, resource_tracker: &ResourceTracker) -> Result<Self, GpuError> {
        let encoder = pool.get_encoder(label, resource_tracker)?;

        Ok(Self {
            encoder: Some(encoder),
            pool,
            operation_id,
        })
    }

    pub fn encoder_mut(&mut self) -> &mut CommandEncoder {
        return self.encoder.as_mut().expect("Encoder already consumed");
    }

    pub fn finish_and_submit(mut self, timeout: Option<Duration>, guard: CommandBufferGuard<'a>) -> Result<(), GpuError> {
        let encoder = self.encoder.take().expect("Encoder already consumed");
        let buffer = encoder.finish();

        self.pool.submit_buffer(buffer, self.operation_id, timeout, guard)
    }
}

impl<'a> Drop for ManagedEncoder<'a> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.pool.return_encoder(encoder);
        }
    }
}

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Success,
    Failed(String),
    TimedOut,
    Cancelled,
}

#[derive(Debug)]
pub struct TaskCompletion {
    pub operation_id: OperationId,
    pub variable_id: VariableId,
    pub status: TaskStatus,
    pub duration: Duration,
    pub gradient_norm: Option<f32>,
}

impl TaskCompletion {
    /// Create a successful task completion
    pub fn success(operation_id: OperationId, variable_id: VariableId, duration: Duration, gradient_norm: Option<f32>) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::Success,
            duration,
            gradient_norm,
        };
    }

    /// Create a failed task completion
    pub fn failed(operation_id: OperationId, variable_id: VariableId, duration: Duration, error: String) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::Failed(error),
            duration,
            gradient_norm: None,
        };
    }

    /// Create a timeout task completion
    pub fn timed_out(operation_id: OperationId, variable_id: VariableId, duration: Duration) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::TimedOut,
            duration,
            gradient_norm: None,
        };
    }

    /// Create a cancelled task completion
    pub fn cancelled(operation_id: OperationId, variable_id: VariableId, duration: Duration) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::Cancelled,
            duration,
            gradient_norm: None,
        };
    }

    pub fn is_success(&self) -> bool {
        return matches!(self.status, TaskStatus::Success);
    }

    pub fn is_failed(&self) -> bool {
        return !self.is_success();
    }

    pub fn is_system_failure(&self) -> bool {
        matches!(self.status, TaskStatus::TimedOut | TaskStatus::Cancelled)
    }

    pub fn get_error_message(&self) -> Option<&str> {
        match &self.status {
            TaskStatus::Failed(msg) => Some(msg),
            TaskStatus::TimedOut => Some("Operation timed out"),
            TaskStatus::Cancelled => Some("Operation was cancelled"),
            TaskStatus::Success => None,
        }
    }

    pub fn has_gradient_issues(&self) -> bool {
        match self.gradient_norm {
            Some(norm) => norm.is_nan() || norm.is_infinite() || norm < 1e-7,
            None => false,
        }
    }

    pub fn from_timed_result<T, E>(operation_id: OperationId, variable_id: VariableId, start_time: Instant, result: Result<T, E>, gradient_norm: Option<f32>) -> Self
    where
        E: std::fmt::Display,
    {
        let duration = start_time.elapsed();

        match result {
            Ok(_) => Self::success(operation_id, variable_id, duration, gradient_norm),
            Err(e) => Self::failed(operation_id, variable_id, duration, e.to_string()),
        }
    }

    pub fn log(&self) {
        match &self.status {
            TaskStatus::Success => {
                if let Some(norm) = self.gradient_norm {
                    if self.has_gradient_issues() {
                        warn!(
                            operation_id = ?self.operation_id,
                            variable_id = ?self.variable_id,
                            duration_ms = self.duration.as_millis(),
                            gradient_norm = norm,
                            "Task completed with gradient issues"
                        );
                    } else {
                        debug!(
                            operation_id = ?self.operation_id,
                            variable_id = ?self.variable_id,
                            duration_ms = self.duration.as_millis(),
                            gradient_norm = norm,
                            "Task completed successfully"
                        );
                    }
                } else {
                    debug!(
                        operation_id = ?self.operation_id,
                        variable_id = ?self.variable_id,
                        duration_ms = self.duration.as_millis(),
                        "Task completed successfully"
                    );
                }
            }
            TaskStatus::Failed(error) => {
                error!(
                    operation_id = ?self.operation_id,
                    variable_id = ?self.variable_id,
                    duration_ms = self.duration.as_millis(),
                    error = error,
                    "Task failed"
                );
            }
            TaskStatus::TimedOut => {
                error!(
                    operation_id = ?self.operation_id,
                    variable_id = ?self.variable_id,
                    duration_ms = self.duration.as_millis(),
                    "Task timed out"
                );
            }
            TaskStatus::Cancelled => {
                warn!(
                    operation_id = ?self.operation_id,
                    variable_id = ?self.variable_id,
                    duration_ms = self.duration.as_millis(),
                    "Task was cancelled"
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed = 0,   // Normal operation
    Open = 1,     // Failing, blocking requests
    HalfOpen = 2, // Testing recovery
}

impl From<u8> for CircuitBreakerState {
    fn from(value: u8) -> Self {
        match value {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitBreakerState,
    pub failure_count: usize,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub rejected_requests: usize,
    pub success_rate: f64,
    pub time_in_current_state: Duration,
}

impl CircuitBreakerStats {
    /// Check if the circuit breaker stats indicate problems
    pub fn indicates_problems(&self) -> bool {
        match self.state {
            CircuitBreakerState::Open => true,
            CircuitBreakerState::HalfOpen => Duration::from_secs(60) < self.time_in_current_state,
            CircuitBreakerState::Closed => (self.success_rate < 0.8) && (10 < self.total_requests),
        }
    }
}

pub struct CircuitBreaker {
    state: AtomicU8,
    total_requests: AtomicUsize,
    successful_requests: AtomicUsize,
    failure_count: AtomicUsize,
    last_failure: AtomicU64,
    failure_threshold: usize,
    rejected_requests: AtomicUsize,
    recovery_timeout: Duration,
    last_state_change: AtomicU64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        return Self {
            state: AtomicU8::new(CircuitBreakerState::Closed as u8),
            total_requests: AtomicUsize::new(0),
            successful_requests: AtomicUsize::new(0),
            failure_count: AtomicUsize::new(0),
            last_failure: AtomicU64::new(0),
            failure_threshold,
            rejected_requests: AtomicUsize::new(0),
            recovery_timeout,
            last_state_change: AtomicU64::new(now),
        };
    }

    pub fn for_gpu_operations() -> Self {
        return Self::new(5, Duration::from_secs(10));
    }

    /// Check if circuit breaker should attempt recovery
    fn should_attempt_recovery(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let last_failure = self.last_failure.load(Ordering::Acquire);
        let time_since_failure = Duration::from_secs(now.saturating_sub(last_failure));

        return self.recovery_timeout <= time_since_failure;
    }

    /// Get current circuit breaker state
    pub fn get_state(&self) -> CircuitBreakerState {
        CircuitBreakerState::from(self.state.load(Ordering::Acquire))
    }

    /// Get time spent in current state
    fn get_time_in_current_state(&self) -> Duration {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let last_change = self.last_state_change.load(Ordering::Relaxed);

        Duration::from_secs(now.saturating_sub(last_change))
    }

    pub fn get_stats(&self) -> CircuitBreakerStats {
        let total = self.total_requests.load(Ordering::Relaxed);
        let successful = self.successful_requests.load(Ordering::Relaxed);
        let rejected = self.rejected_requests.load(Ordering::Relaxed);
        let success_rate;

        if 0 < total {
            success_rate = (successful as f64) / (total as f64)
        } else {
            success_rate = 0.0
        };

        return CircuitBreakerStats {
            state: self.get_state(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            total_requests: total,
            successful_requests: successful,
            rejected_requests: rejected,
            success_rate,
            time_in_current_state: self.get_time_in_current_state(),
        };
    }

    /// Update the last state change timestamp
    fn update_state_change_time(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.last_state_change.store(now, Ordering::Relaxed);
    }

    fn open(&self) {
        self.state.store(CircuitBreakerState::Open as u8, Ordering::Release);
        self.update_state_change_time();
    }

    fn open_half(&self) {
        self.state.store(CircuitBreakerState::HalfOpen as u8, Ordering::Release);
        self.update_state_change_time();

        tracing::info!("Circuit breaker transitioning to HALF_OPEN");
    }

    fn close(&self) {
        self.state.store(CircuitBreakerState::Closed as u8, Ordering::Release);
        self.failure_count.store(0, Ordering::Relaxed);
        self.update_state_change_time();
    }

    pub fn force_open(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.last_failure.store(now, Ordering::Relaxed);
        self.open();

        tracing::error!("Circuit breaker FORCE OPENED");
    }

    pub fn force_close(&self) {
        self.close();

        tracing::info!("Circuit breaker FORCE CLOSED");
    }

    pub fn is_request_allowed(&self) -> bool {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let current_state = self.get_state();

        match current_state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if self.should_attempt_recovery() {
                    self.open_half();

                    return true;
                } else {
                    self.rejected_requests.fetch_add(1, Ordering::Relaxed);

                    return false;
                }
            }
            CircuitBreakerState::HalfOpen => {
                return true;
            }
        }
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);

        let current_state = self.get_state();

        match current_state {
            CircuitBreakerState::Closed => {
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitBreakerState::HalfOpen => {
                self.close();

                tracing::info!("Circuit breaker recovered, transitioning to CLOSED state");
            }
            CircuitBreakerState::Open => {
                tracing::warn!("Received success in OPEN state, this shouldn't happen");

                self.close();
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.last_failure.store(now, Ordering::Relaxed);

        let new_failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        let current_state = self.get_state();

        match current_state {
            CircuitBreakerState::Closed => {
                if new_failure_count >= self.failure_threshold {
                    self.open();

                    tracing::error!(
                        failure_count = new_failure_count,
                        threshold = self.failure_threshold,
                        "Circuit breaker OPENED due to consecutive failures"
                    );
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.open();

                tracing::warn!("Circuit breaker recovery failed, returning to OPEN state");
            }
            CircuitBreakerState::Open => {}
        }
    }

    pub fn is_healthy(&self) -> bool {
        matches!(self.get_state(), CircuitBreakerState::Closed)
    }
}

#[derive(Debug, Clone)]
pub enum GradientTask {
    // be lightweight and contain only data needed for the specific operation
    Add {
        output_grad: Tensor,
        target_variable: VariableId,
        operation_id: OperationId,
    },
    Mul {
        output_grad: Tensor,
        other_input: Tensor,
        target_variable: VariableId,
        operation_id: OperationId,
    },
    MatMul {
        output_grad: Tensor,
        other_input: Tensor,
        transpose_other: bool,
        target_variable: VariableId,
        operation_id: OperationId,
    },
    Transpose {
        output_grad: Tensor,
        target_variable: VariableId,
        operation_id: OperationId,
    },
}

impl GradientTask {
    /// Get the operation ID for this task
    pub fn get_operation_id(&self) -> OperationId {
        match self {
            GradientTask::Add { operation_id, .. } => *operation_id,
            GradientTask::Mul { operation_id, .. } => *operation_id,
            GradientTask::MatMul { operation_id, .. } => *operation_id,
            GradientTask::Transpose { operation_id, .. } => *operation_id,
        }
    }

    /// Get the target variable ID for this task
    pub fn get_target_variable(&self) -> VariableId {
        match self {
            GradientTask::Add { target_variable, .. } => *target_variable,
            GradientTask::Mul { target_variable, .. } => *target_variable,
            GradientTask::MatMul { target_variable, .. } => *target_variable,
            GradientTask::Transpose { target_variable, .. } => *target_variable,
        }
    }

    /// Get a human-readable name for the task type
    pub fn get_task_type(&self) -> &'static str {
        match self {
            GradientTask::Add { .. } => "Add",
            GradientTask::Mul { .. } => "Mul",
            GradientTask::MatMul { .. } => "MatMul",
            GradientTask::Transpose { .. } => "Transpose",
        }
    }

    /// Estimate the computational complexity of this task
    pub fn estimate_complexity(&self) -> usize {
        match self {
            GradientTask::Add { output_grad, .. } => output_grad.shape.dims.iter().product(),
            GradientTask::Mul { output_grad, .. } => output_grad.shape.dims.iter().product(),
            GradientTask::MatMul { output_grad, other_input, .. } => {
                let m = output_grad.shape.dims.get(0).unwrap_or(&1);
                let n = other_input.shape.dims.get(1).unwrap_or(&1);
                let k = other_input.shape.dims.get(0).unwrap_or(&1);

                m * n * k
            }
            GradientTask::Transpose { output_grad, .. } => output_grad.shape.dims.iter().product(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskManagerStats {
    pub active_tasks: usize,
    pub max_concurrent_tasks: usize,
    pub total_submitted: usize,
    pub total_completed: usize,
    pub total_failed: usize,
    pub queue_length: usize,
    pub available_permits: usize,
    pub circuit_breaker_stats: CircuitBreakerStats,
    pub is_shutdown: bool,
}

impl TaskManagerStats {
    /// Check if stats indicate problems
    pub fn indicates_problems(&self) -> bool {
        let failure_rate = if self.total_submitted > 0 { self.total_failed as f64 / self.total_submitted as f64 } else { 0.0 };

        return self.is_shutdown || self.circuit_breaker_stats.indicates_problems() || failure_rate > 0.1 || (self.queue_length > 1000 && self.active_tasks == self.max_concurrent_tasks);
    }
}

struct TaskGuard {
    active_tasks: Arc<AtomicUsize>,
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.active_tasks.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct GradientTaskManager {
    pub task_queue: flume::Receiver<GradientTask>,
    pub task_sender: flume::Sender<GradientTask>,
    pub active_tasks: Arc<AtomicUsize>,
    pub max_concurrent_tasks: usize,
    pub completion_notifier: watch::Sender<TaskCompletion>,
    pub circuit_breaker: Arc<CircuitBreaker>,

    semaphore: Arc<Semaphore>,
    shutdown_signal: Arc<AtomicBool>,
    task_timeout: Duration,

    total_tasks_submitted: AtomicUsize,
    total_tasks_completed: AtomicUsize,
    total_tasks_failed: AtomicUsize,
}

impl GradientTaskManager {
    /// Create a new gradient task manager
    pub fn new(max_concurrent_tasks: usize, circuit_breaker: CircuitBreaker, task_timeout: Duration) -> (Self, watch::Receiver<TaskCompletion>) {
        let (task_sender, task_queue) = flume::unbounded();
        let (completion_notifier, completion_receiver) = watch::channel(TaskCompletion::success(OperationId(0), VariableId(0), Duration::from_millis(0), None));

        return (
            Self {
                task_queue,
                task_sender,
                active_tasks: Arc::new(AtomicUsize::new(0)),
                max_concurrent_tasks,
                completion_notifier,
                circuit_breaker: Arc::new(circuit_breaker),
                semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
                shutdown_signal: Arc::new(AtomicBool::new(false)),
                task_timeout,
                total_tasks_submitted: AtomicUsize::new(0),
                total_tasks_completed: AtomicUsize::new(0),
                total_tasks_failed: AtomicUsize::new(0),
            },
            completion_receiver,
        );
    }

    /// Create a task manager optimized for GPU gradient computation
    pub fn for_gpu_gradients(max_concurrent_tasks: usize) -> (Self, watch::Receiver<TaskCompletion>) {
        Self::new(max_concurrent_tasks, CircuitBreaker::for_gpu_operations(), Duration::from_secs(10))
    }

    /// Submit a gradient task for execution
    pub async fn submit_task(&self, task: GradientTask) -> Result<(), String> {
        if self.shutdown_signal.load(Ordering::Acquire) == true {
            return Err("Task manager is shutting down".to_string());
        }

        if self.circuit_breaker.is_request_allowed() == false {
            let completion = TaskCompletion::failed(task.get_operation_id(), task.get_target_variable(), Duration::from_millis(0), "Circuit breaker is open".to_string());
            let _ = self.completion_notifier.send(completion);

            return Err("Circuit breaker is open, rejecting task".to_string());
        }

        self.total_tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.task_sender.send_async(task).await.map_err(|_| "Failed to queue task, manager may be shutting down".to_string())?;

        return Ok(());
    }

    pub async fn run(&self, resource_tracker: Arc<ResourceTracker>) {
        info!("Starting gradient task manager with {} max concurrent tasks", self.max_concurrent_tasks);

        loop {
            if self.shutdown_signal.load(Ordering::Acquire) {
                info!("Shutdown signal received, stopping task manager");
                break;
            }

            let task = match self.task_queue.recv_async().await {
                Ok(task) => task,
                Err(_) => {
                    warn!("Task queue closed, stopping task manager");
                    break;
                }
            };

            let permit = match self.semaphore.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    error!("Failed to acquire semaphore permit");
                    continue;
                }
            };

            let resource_tracker = resource_tracker.clone();
            let active_tasks = self.active_tasks.clone();
            let completion_notifier = self.completion_notifier.clone();
            let circuit_breaker = Arc::clone(&self.circuit_breaker);
            let task_timeout = self.task_timeout;
            let shutdown_signal = self.shutdown_signal.clone();

            tokio::spawn(async move {
                active_tasks.fetch_add(1, Ordering::Relaxed);

                let _ = TaskGuard { active_tasks: active_tasks.clone() };
                let start_time = Instant::now();
                let result = timeout(task_timeout, Self::execute_gradient_task(task.clone(), resource_tracker, shutdown_signal)).await;

                let completion = match result {
                    Ok(Ok(gradient_norm)) => {
                        circuit_breaker.record_success();
                        TaskCompletion::success(task.get_operation_id(), task.get_target_variable(), start_time.elapsed(), gradient_norm)
                    }
                    Ok(Err(error)) => {
                        circuit_breaker.record_failure();
                        TaskCompletion::failed(task.get_operation_id(), task.get_target_variable(), start_time.elapsed(), error)
                    }
                    Err(_) => {
                        circuit_breaker.record_failure();
                        TaskCompletion::timed_out(task.get_operation_id(), task.get_target_variable(), start_time.elapsed())
                    }
                };

                completion.log();

                let _ = completion_notifier.send(completion);

                drop(permit);
            });
        }

        while 0 < self.active_tasks.load(Ordering::Acquire) {
            sleep(Duration::from_millis(100)).await;
        }

        info!("Gradient task manager stopped");
    }

    /// Compute gradient norm for monitoring
    fn compute_gradient_norm(gradient: &Tensor) -> f32 {
        let sum_squares: f32 = gradient.data.iter().map(|x| x * x).sum();

        return (sum_squares / gradient.data.len() as f32).sqrt();
    }

    /// Execute a gradient task
    async fn execute_gradient_task(task: GradientTask, resource_tracker: Arc<ResourceTracker>, shutdown_signal: Arc<AtomicBool>) -> Result<Option<f32>, String> {
        if shutdown_signal.load(Ordering::Acquire) == true {
            return Err("Shutdown requested".to_string());
        }

        let _ = resource_tracker.reserve_operation().map_err(|e| format!("Failed to reserve GPU resources: {}", e))?;

        match task {
            GradientTask::Add { output_grad, target_variable, .. } => {
                let norm = Self::compute_gradient_norm(&output_grad);

                info!(target_variable = ?target_variable, gradient_norm = norm, "Computed Add gradient");

                Ok(Some(norm))
            }
            GradientTask::Mul {
                output_grad,
                other_input,
                target_variable,
                ..
            } => {
                let norm = Self::compute_gradient_norm(&output_grad);

                info!(target_variable = ?target_variable, gradient_norm = norm, "Computed Mul gradient");

                Ok(Some(norm))
            }
            GradientTask::MatMul {
                output_grad,
                other_input,
                transpose_other,
                target_variable,
                ..
            } => {
                let norm = Self::compute_gradient_norm(&output_grad);

                info!(
                    target_variable = ?target_variable,
                    gradient_norm = norm,
                    transpose_other = transpose_other,
                    "Computed MatMul gradient"
                );

                Ok(Some(norm))
            }
            GradientTask::Transpose { output_grad, target_variable, .. } => {
                let norm = Self::compute_gradient_norm(&output_grad);

                info!(target_variable = ?target_variable, gradient_norm = norm, "Computed Transpose gradient");

                Ok(Some(norm))
            }
        }
    }

    /// Shutdown the task manager
    pub async fn shutdown(&self) {
        info!("Initiating task manager shutdown");

        self.shutdown_signal.store(true, Ordering::Release);

        let _ = &self.task_sender;
        let shutdown_timeout = Duration::from_secs(30);
        let start = Instant::now();

        while self.active_tasks.load(Ordering::Acquire) > 0 {
            if start.elapsed() > shutdown_timeout {
                warn!("Shutdown timeout reached, some tasks may not have completed");
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }

        info!("Task manager shutdown complete");
    }

    /// Force emergency shutdown
    pub fn emergency_shutdown(&self) {
        error!("Emergency shutdown triggered");

        self.shutdown_signal.store(true, Ordering::Release);
        self.circuit_breaker.force_open();
    }

    /// Get current task manager statistics
    pub fn get_stats(&self) -> TaskManagerStats {
        return TaskManagerStats {
            active_tasks: self.active_tasks.load(Ordering::Acquire),
            max_concurrent_tasks: self.max_concurrent_tasks,
            total_submitted: self.total_tasks_submitted.load(Ordering::Acquire),
            total_completed: self.total_tasks_completed.load(Ordering::Acquire),
            total_failed: self.total_tasks_failed.load(Ordering::Acquire),
            queue_length: self.task_queue.len(),
            available_permits: self.semaphore.available_permits(),
            circuit_breaker_stats: self.circuit_breaker.get_stats(),
            is_shutdown: self.shutdown_signal.load(Ordering::Acquire),
        };
    }

    /// Check if the task manager is healthy
    pub fn is_healthy(&self) -> bool {
        return (self.shutdown_signal.load(Ordering::Acquire) == false) && (self.circuit_breaker.is_healthy() == true) && (self.active_tasks.load(Ordering::Acquire) < self.max_concurrent_tasks);
    }
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    Direct,    // Allocate immediately when needed
    Pooled,    // Use pre-allocated buffer pool
    Lazy,      // Allocate on first use
    Streaming, // For large sequential operations
}

impl AllocationStrategy {
    /// Get the strategy best suited for a given buffer size and usage pattern
    pub fn choose_for_usage(size_bytes: usize, usage: BufferUsages, is_temporary: bool) -> Self {
        match (size_bytes, is_temporary) {
            (s, _) if s < 1024 * 1024 => Self::Pooled,
            (s, true) if 16 * 1024 * 1024 < s => Self::Direct,
            (s, _) if 64 * 1024 * 1024 < s && usage.contains(BufferUsages::MAP_READ | BufferUsages::MAP_WRITE) => Self::Streaming,

            _ => Self::Lazy,
        }
    }

    /// Check if this strategy supports buffer reuse
    pub fn supports_reuse(&self) -> bool {
        matches!(self, Self::Pooled | Self::Lazy)
    }

    /// Get the cleanup priority (higher = clean up sooner)
    pub fn cleanup_priority(&self) -> u8 {
        match self {
            Self::Direct => 255,
            Self::Streaming => 200,
            Self::Lazy => 100,
            Self::Pooled => 50,
        }
    }
}

pub struct BufferInfo {
    buffer: Arc<Buffer>,
    size: usize,
    usage: BufferUsages,
    strategy: AllocationStrategy,
    created_at: Instant,
    last_used: AtomicU64, // Unix timestamp in seconds
    ref_count: AtomicUsize,
    is_pooled: bool,
}

impl BufferInfo {
    pub fn new(buffer: Buffer, size: usize, usage: BufferUsages, strategy: AllocationStrategy, is_pooled: bool) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        return Self {
            buffer: Arc::new(buffer),
            size,
            usage,
            strategy,
            created_at: Instant::now(),
            last_used: AtomicU64::new(now),
            ref_count: AtomicUsize::new(1),
            is_pooled,
        };
    }

    pub fn touch(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.last_used.store(now, Ordering::Relaxed);
    }

    pub fn age(&self) -> Duration {
        return self.created_at.elapsed();
    }

    pub fn seconds_since_last_use(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let last_used = self.last_used.load(Ordering::Relaxed);

        return now.saturating_sub(last_used);
    }

    pub fn increment_ref(&self) -> usize {
        return self.ref_count.fetch_add(1, Ordering::Relaxed) + 1;
    }

    pub fn decrement_ref(&self) -> usize {
        let prev = self.ref_count.fetch_sub(1, Ordering::Relaxed);

        return prev.saturating_sub(1);
    }

    pub fn get_ref_count(&self) -> usize {
        return self.ref_count.load(Ordering::Relaxed);
    }

    /// Check if this buffer is a candidate for cleanup
    pub fn is_cleanup_candidate(&self, max_age_seconds: u64, min_idle_seconds: u64) -> bool {
        if 1 < self.get_ref_count() {
            return false;
        }

        let age_seconds = self.age().as_secs();
        let idle_seconds = self.seconds_since_last_use();

        if self.is_pooled && age_seconds < max_age_seconds / 2 {
            return false;
        }

        return age_seconds > max_age_seconds || idle_seconds > min_idle_seconds;
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated_bytes: usize,
    pub peak_usage_bytes: usize,
    pub active_buffers: usize,
    pub pooled_buffers: usize,
    cleanup_queue_length: usize,
    allocation_strategy_counts: [usize; 4], // [Direct, Pooled, Lazy, Streaming]
}

impl MemoryStats {
    pub fn allocation_efficiency(&self) -> f64 {
        if self.active_buffers == 0 {
            return 1.0;
        }

        if (self.active_buffers <= 2) && (self.pooled_buffers == 0) {
            return 0.5;
        }

        let pooled_ratio = self.pooled_buffers as f64 / self.active_buffers as f64;

        return (pooled_ratio * 0.8) + 0.2;
    }

    pub fn memory_pressure(&self) -> f64 {
        if self.peak_usage_bytes == 0 {
            return 0.0;
        }

        return (self.total_allocated_bytes as f64) / (self.peak_usage_bytes as f64);
    }
}

pub struct BufferPool {
    available_buffers: VecDeque<BufferId>,
    size_class: usize, // Buffers in this pool are approximately this size
    usage: BufferUsages,
    max_pool_size: usize,
    hit_count: AtomicUsize,
    miss_count: AtomicUsize,
}

impl BufferPool {
    pub fn new(size_class: usize, usage: BufferUsages, max_pool_size: usize) -> Self {
        Self {
            available_buffers: VecDeque::with_capacity(max_pool_size),
            size_class,
            usage,
            max_pool_size,
            hit_count: AtomicUsize::new(0),
            miss_count: AtomicUsize::new(0),
        }
    }

    pub fn try_get_buffer(&mut self, requested_size: usize) -> Option<BufferId> {
        let size_tolerance = self.size_class / 4;

        if (requested_size < self.size_class.saturating_sub(size_tolerance)) || (self.size_class + size_tolerance < requested_size) {
            return None;
        }

        if let Some(buffer_id) = self.available_buffers.pop_front() {
            self.hit_count.fetch_add(1, Ordering::Relaxed);

            Some(buffer_id)
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);

            return None;
        }
    }

    pub fn return_buffer(&mut self, buffer_id: BufferId) -> bool {
        if self.max_pool_size <= self.available_buffers.len() {
            return false;
        }

        self.available_buffers.push_back(buffer_id);

        return true;
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            return 0.0;
        } else {
            return (hits as f64) / (total as f64);
        }
    }

    pub fn len(&self) -> usize {
        return self.available_buffers.len();
    }

    pub fn clear(&mut self) -> usize {
        let count = self.available_buffers.len();

        self.available_buffers.clear();

        return count;
    }
}

pub struct GpuMemoryManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    resource_tracker: Arc<ResourceTracker>,

    allocated_buffers: DashMap<BufferId, BufferInfo>,
    memory_usage: AtomicUsize,
    peak_usage: AtomicUsize,

    buffer_pools: Mutex<Vec<BufferPool>>,

    cleanup_scheduler: mpsc::UnboundedSender<BufferId>,
    cleanup_receiver: Mutex<mpsc::UnboundedReceiver<BufferId>>,

    allocation_strategy: AllocationStrategy,
    max_memory_bytes: usize,
    cleanup_interval: Duration,

    total_allocations: AtomicUsize,
    total_deallocations: AtomicUsize,
    cleanup_runs: AtomicUsize,
    emergency_cleanups: AtomicUsize,
}

impl GpuMemoryManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, resource_tracker: Arc<ResourceTracker>, allocation_strategy: AllocationStrategy, max_memory_mb: usize) -> Self {
        let (cleanup_sender, cleanup_receiver) = mpsc::unbounded_channel();
        let mut pools = Vec::new();

        for &size_class in &[
            1024,             // 1KB - small uniforms
            4 * 1024,         // 4KB - small buffers
            64 * 1024,        // 64KB - medium buffers
            1024 * 1024,      // 1MB - large buffers
            16 * 1024 * 1024, // 16MB - very large buffers
        ] {
            pools.push(BufferPool::new(size_class, BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC, 8));
        }

        return Self {
            device,
            queue,
            resource_tracker,
            allocated_buffers: DashMap::new(),
            memory_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            buffer_pools: Mutex::new(pools),
            cleanup_scheduler: cleanup_sender,
            cleanup_receiver: Mutex::new(cleanup_receiver),
            allocation_strategy,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            cleanup_interval: Duration::from_secs(30),
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
            cleanup_runs: AtomicUsize::new(0),
            emergency_cleanups: AtomicUsize::new(0),
        };
    }

    async fn try_get_from_pool(&self, size_bytes: usize, usage: BufferUsages) -> Option<BufferId> {
        let mut pools = self.buffer_pools.lock().await;

        for pool in pools.iter_mut() {
            if pool.usage.contains(usage) {
                if let Some(buffer_id) = pool.try_get_buffer(size_bytes) {
                    if let Some(entry) = self.allocated_buffers.get(&buffer_id) {
                        let buffer_info = entry.value();

                        buffer_info.touch();
                        buffer_info.increment_ref();

                        debug!(
                            buffer_id = ?buffer_id,
                            size_bytes = size_bytes,
                            pool_hit_rate = pool.hit_rate(),
                            "Retrieved buffer from pool"
                        );

                        return Some(buffer_id);
                    } else {
                        warn!(buffer_id = ?buffer_id, "Buffer in pool but not in allocated_buffers");
                    }
                }
            }
        }

        return None;
    }

    /// Allocate a new GPU buffer with the specified parameters
    pub async fn allocate_buffer(&self, size_bytes: usize, usage: BufferUsages, label: Option<&str>, strategy_override: Option<AllocationStrategy>) -> Result<BufferId, GpuError> {
        let current_usage = self.memory_usage.load(Ordering::Acquire);

        if self.max_memory_bytes < current_usage + size_bytes {
            self.try_emergency_cleanup().await;

            let current_usage = self.memory_usage.load(Ordering::Acquire);

            if self.max_memory_bytes < current_usage + size_bytes {
                return Err(GpuError::MemoryExhaustion {
                    requested: size_bytes / (1024 * 1024),
                    available: (self.max_memory_bytes - current_usage) / (1024 * 1024),
                });
            }
        }

        let strategy = strategy_override.unwrap_or_else(|| AllocationStrategy::choose_for_usage(size_bytes, usage, false));

        if matches!(strategy, AllocationStrategy::Pooled) {
            if let Some(buffer_id) = self.try_get_from_pool(size_bytes, usage).await {
                return Ok(buffer_id);
            }
        }

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label,
            size: size_bytes as u64,
            usage,
            mapped_at_creation: false,
        });

        let buffer_id = next_buffer_id();
        let is_pooled = matches!(strategy, AllocationStrategy::Pooled);
        let buffer_info = BufferInfo::new(buffer, size_bytes, usage, strategy, is_pooled);

        self.memory_usage.fetch_add(size_bytes, Ordering::Relaxed);

        let new_usage = self.memory_usage.load(Ordering::Relaxed);

        loop {
            let current_peak = self.peak_usage.load(Ordering::Relaxed);

            if new_usage <= current_peak {
                break;
            }

            if self.peak_usage.compare_exchange_weak(current_peak, new_usage, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }

        self.allocated_buffers.insert(buffer_id, buffer_info);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        Ok(buffer_id)
    }

    /// Get a reference to a buffer, incrementing its reference count
    pub fn get_buffer(&self, buffer_id: BufferId) -> Option<Arc<Buffer>> {
        self.allocated_buffers.get(&buffer_id).map(|entry| {
            let buffer_info = entry.value();

            buffer_info.touch();
            buffer_info.increment_ref();
            buffer_info.buffer.clone()
        })
    }

    /// Release a buffer reference, potentially scheduling it for cleanup
    pub async fn release_buffer(&self, buffer_id: BufferId) -> Result<(), GpuError> {
        if let Some(entry) = self.allocated_buffers.get(&buffer_id) {
            let buffer_info = entry.value();
            let new_ref_count = buffer_info.decrement_ref();

            if new_ref_count == 0 {
                if let Err(_) = self.cleanup_scheduler.send(buffer_id) {
                    warn!(buffer_id = ?buffer_id, "Failed to schedule buffer cleanup");
                }
            }

            debug!(
                buffer_id = ?buffer_id,
                ref_count = new_ref_count,
                "Released buffer reference"
            );
        }

        Ok(())
    }

    /// Try to return a buffer to the pool
    async fn try_return_to_pool(&self, buffer_id: BufferId) -> bool {
        if let Some(entry) = self.allocated_buffers.get(&buffer_id) {
            let buffer_info = entry.value();

            if !buffer_info.is_pooled {
                return false;
            }

            let mut pools = self.buffer_pools.lock().await;

            for pool in pools.iter_mut() {
                if pool.usage.contains(buffer_info.usage) {
                    if pool.return_buffer(buffer_id) {
                        debug!(
                            buffer_id = ?buffer_id,
                            size_bytes = buffer_info.size,
                            "Returned buffer to pool"
                        );
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /// Clean up a specific buffer
    async fn cleanup_buffer(&self, buffer_id: BufferId) {
        if self.try_return_to_pool(buffer_id).await == true {
            debug!(buffer_id = ?buffer_id, "Buffer returned to pool for reuse");
        } else {
            if let Some((_, buffer_info)) = self.allocated_buffers.remove(&buffer_id) {
                self.memory_usage.fetch_sub(buffer_info.size, Ordering::Relaxed);
                self.total_deallocations.fetch_add(1, Ordering::Relaxed);

                debug!(
                    buffer_id = ?buffer_id,
                    size_bytes = buffer_info.size,
                    age_seconds = buffer_info.age().as_secs(),
                    "Deallocated GPU buffer"
                );
            }
        }
    }

    /// Periodic cleanup of old/unused buffers
    async fn periodic_cleanup(&self) {
        let start = Instant::now();
        let mut cleaned = 0;
        let cleanup_candidates: Vec<BufferId> = self
            .allocated_buffers
            .iter()
            .filter_map(|entry| {
                let buffer_id = *entry.key();
                let buffer_info = entry.value();
                let should_cleanup = match buffer_info.strategy {
                    AllocationStrategy::Direct => {
                        buffer_info.is_cleanup_candidate(300, 60) // 5 min max age, 1 min idle
                    }
                    AllocationStrategy::Pooled => {
                        buffer_info.is_cleanup_candidate(1800, 300) // 30 min max age, 5 min idle
                    }
                    AllocationStrategy::Lazy => {
                        buffer_info.is_cleanup_candidate(900, 180) // 15 min max age, 3 min idle
                    }
                    AllocationStrategy::Streaming => {
                        buffer_info.is_cleanup_candidate(60, 30) // 1 min max age, 30 sec idle
                    }
                };

                if should_cleanup {
                    Some(buffer_id)
                } else {
                    None
                }
            })
            .collect();

        for buffer_id in cleanup_candidates {
            self.cleanup_buffer(buffer_id).await;

            cleaned += 1;
        }

        if 0 < cleaned {
            let duration = start.elapsed();

            info!(
                cleaned_buffers = cleaned,
                cleanup_duration_ms = duration.as_millis(),
                current_memory_mb = self.memory_usage.load(Ordering::Relaxed) / (1024 * 1024),
                "Completed periodic cleanup"
            );
        }

        self.cleanup_runs.fetch_add(1, Ordering::Relaxed);
    }

    /// Run the cleanup background task
    pub async fn run_cleanup_task(&self) {
        info!("Starting GPU memory cleanup task");

        let mut cleanup_receiver = self.cleanup_receiver.lock().await;
        let mut cleanup_timer = tokio::time::interval(self.cleanup_interval);

        loop {
            tokio::select! {
                buffer_id = cleanup_receiver.recv() => {
                    if let Some(buffer_id) = buffer_id {
                        self.cleanup_buffer(buffer_id).await;
                    } else {
                        info!("Cleanup channel closed, stopping cleanup task");
                        break;
                    }
                }

                _ = cleanup_timer.tick() => {
                    self.periodic_cleanup().await;
                }
            }
        }
    }

    /// Emergency cleanup when memory is running low
    async fn try_emergency_cleanup(&self) {
        warn!("Running emergency memory cleanup");

        let start = Instant::now();
        let mut cleaned = 0;
        let emergency_candidates: Vec<BufferId> = self
            .allocated_buffers
            .iter()
            .filter_map(|entry| {
                let buffer_id = *entry.key();
                let buffer_info = entry.value();

                if buffer_info.get_ref_count() <= 1 {
                    Some(buffer_id)
                } else {
                    None
                }
            })
            .collect();

        for buffer_id in emergency_candidates {
            self.cleanup_buffer(buffer_id).await;
            cleaned += 1;
        }

        let duration = start.elapsed();

        error!(
            cleaned_buffers = cleaned,
            cleanup_duration_ms = duration.as_millis(),
            current_memory_mb = self.memory_usage.load(Ordering::Relaxed) / (1024 * 1024),
            "Completed emergency cleanup"
        );

        self.emergency_cleanups.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let mut strategy_counts = [0; 4];
        let mut pooled_count = 0;

        for entry in self.allocated_buffers.iter() {
            let buffer_info = entry.value();
            let strategy_index = match buffer_info.strategy {
                AllocationStrategy::Direct => 0,
                AllocationStrategy::Pooled => 1,
                AllocationStrategy::Lazy => 2,
                AllocationStrategy::Streaming => 3,
            };

            strategy_counts[strategy_index] += 1;

            if buffer_info.is_pooled == true {
                pooled_count += 1;
            }
        }

        return MemoryStats {
            total_allocated_bytes: self.memory_usage.load(Ordering::Relaxed),
            peak_usage_bytes: self.peak_usage.load(Ordering::Relaxed),
            active_buffers: self.allocated_buffers.len(),
            pooled_buffers: pooled_count,
            cleanup_queue_length: 0,
            allocation_strategy_counts: strategy_counts,
        };
    }

    /// Force cleanup of all buffers (for shutdown)
    pub async fn shutdown(&self) {
        info!("Shutting down GPU memory manager");

        let mut pools = self.buffer_pools.lock().await;
        let mut total_cleared = 0;

        for pool in pools.iter_mut() {
            while let Some(buffer_id) = pool.available_buffers.pop_front() {
                self.allocated_buffers.remove(&buffer_id);
                total_cleared += 1;
            }
        }

        drop(pools);

        let buffer_ids: Vec<BufferId> = self.allocated_buffers.iter().map(|entry| *entry.key()).collect();

        for buffer_id in buffer_ids {
            if let Some((_, _)) = self.allocated_buffers.remove(&buffer_id) {}
        }

        info!(
            cleared_pool_buffers = total_cleared,
            final_memory_usage = self.memory_usage.load(Ordering::Relaxed),
            "GPU memory manager shutdown complete"
        );
    }

    pub fn is_healthy(&self) -> bool {
        let stats = self.get_stats();
        let memory_pressure = stats.memory_pressure();
        let allocation_efficiency = stats.allocation_efficiency();

        return (memory_pressure <= 1.0) && (0.3 < allocation_efficiency);
    }
}

pub struct GpuMetrics {
    operations_completed: AtomicU64,
    operations_failed: AtomicU64,
    average_latency_ms: AtomicU64,
    memory_usage_mb: AtomicUsize,
    queue_depth: AtomicUsize,
    pub last_heartbeat: AtomicU64,
}

impl GpuMetrics {
    pub fn new() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        return Self {
            operations_completed: AtomicU64::new(0),
            operations_failed: AtomicU64::new(0),
            average_latency_ms: AtomicU64::new(0),
            memory_usage_mb: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            last_heartbeat: AtomicU64::new(now),
        };
    }

    fn update_average_latency(&self, new_duration: Duration) {
        let new_latency_ms = new_duration.as_millis() as u64;
        let current_avg = self.average_latency_ms.load(Ordering::Relaxed);
        let new_avg;

        if current_avg == 0 {
            new_avg = new_latency_ms
        } else {
            new_avg = ((current_avg as f64 * 0.9) + (new_latency_ms as f64 * 0.1)) as u64
        };

        self.average_latency_ms.store(new_avg, Ordering::Relaxed);
    }

    /// Update memory usage
    fn update_memory_usage(&self, usage_mb: usize) {
        self.memory_usage_mb.store(usage_mb, Ordering::Relaxed);
    }

    /// Update queue depth
    fn update_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Update heartbeat timestamp
    pub fn heartbeat(&self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        self.last_heartbeat.store(now, Ordering::Relaxed);
    }

    /// Record a completed operation with its execution time
    pub fn record_operation_completed(&self, duration: Duration) {
        self.operations_completed.fetch_add(1, Ordering::Relaxed);
        self.update_average_latency(duration);
        self.heartbeat();
    }

    /// Record a failed operation
    pub fn record_operation_failed(&self, duration: Duration) {
        self.operations_failed.fetch_add(1, Ordering::Relaxed);
        self.update_average_latency(duration);
        self.heartbeat();
    }

    /// Get current success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let completed = self.operations_completed.load(Ordering::Relaxed);
        let failed = self.operations_failed.load(Ordering::Relaxed);
        let total = completed + failed;

        if total == 0 {
            return 1.0;
        } else {
            return (completed as f64) / (total as f64);
        }
    }

    /// Get total operations processed
    pub fn total_operations(&self) -> u64 {
        return self.operations_completed.load(Ordering::Relaxed) + self.operations_failed.load(Ordering::Relaxed);
    }

    /// Get average latency in milliseconds
    pub fn average_latency_ms(&self) -> u64 {
        return self.average_latency_ms.load(Ordering::Relaxed);
    }

    /// Get current memory usage in MB
    pub fn memory_usage_mb(&self) -> usize {
        return self.memory_usage_mb.load(Ordering::Relaxed);
    }

    /// Get current queue depth
    pub fn queue_depth(&self) -> usize {
        return self.queue_depth.load(Ordering::Relaxed);
    }

    /// Check if the system is responsive (heartbeat within last 30 seconds)
    pub fn is_responsive(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);
        let age_sec = now.saturating_sub(last_heartbeat);

        return age_sec <= 30;
    }

    /// Get time since last heartbeat in seconds
    pub fn seconds_since_heartbeat(&self) -> u64 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);

        return now.saturating_sub(last_heartbeat);
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub success_rate: f64,
    pub average_compute_time_ms: f64,
    pub operations_per_second: f64,
    pub peak_memory_usage_mb: usize,
    pub average_queue_depth: usize,
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

impl PerformanceCounter {
    pub fn new() -> Self {
        return Self {
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            total_compute_time_ms: AtomicU64::new(0),
            peak_memory_usage_mb: AtomicUsize::new(0),
            average_queue_depth: AtomicUsize::new(0),
        };
    }

    /// Update peak memory usage if current usage is higher
    pub fn update_peak_memory(&self, current_usage_mb: usize) {
        loop {
            let current_peak = self.peak_memory_usage_mb.load(Ordering::Relaxed);

            if current_usage_mb <= current_peak {
                break;
            }

            if self.peak_memory_usage_mb.compare_exchange_weak(current_peak, current_usage_mb, Ordering::Relaxed, Ordering::Relaxed).is_ok() == true {
                break;
            }
        }
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);

        if total == 0 {
            return 1.0;
        } else {
            let successful = self.successful_operations.load(Ordering::Relaxed);

            return (successful as f64) / (total as f64);
        }
    }

    /// Get average compute time per operation in milliseconds
    pub fn average_compute_time_ms(&self) -> f64 {
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        let total_time = self.total_compute_time_ms.load(Ordering::Relaxed);

        if total_ops == 0 {
            return 0.0;
        } else {
            return (total_time as f64) / (total_ops as f64);
        }
    }

    /// Get operations per second based on total compute time
    pub fn operations_per_second(&self) -> f64 {
        let total_ops = self.total_operations.load(Ordering::Relaxed);
        let total_time_ms = self.total_compute_time_ms.load(Ordering::Relaxed);

        if total_time_ms == 0 {
            return 0.0;
        } else {
            return (total_ops as f64) / ((total_time_ms as f64) / 1000.0);
        }
    }

    pub fn update_queue_depth(&self, current_depth: usize) {
        let avg_current = self.average_queue_depth.load(Ordering::Relaxed);
        let avg_new;

        if avg_current == 0 {
            avg_new = current_depth
        } else {
            avg_new = ((avg_current as f64 * 0.9) + (current_depth as f64 * 0.1)) as usize
        };

        self.average_queue_depth.store(avg_new, Ordering::Relaxed);
    }

    /// Record a successful operation
    pub fn record_success(&self, compute_time: Duration) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.successful_operations.fetch_add(1, Ordering::Relaxed);
        self.total_compute_time_ms.fetch_add(compute_time.as_millis() as u64, Ordering::Relaxed);
    }

    /// Record a failed operation
    pub fn record_failed(&self, compute_time: Duration) {
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
        self.total_compute_time_ms.fetch_add(compute_time.as_millis() as u64, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> PerformanceStats {
        PerformanceStats {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            successful_operations: self.successful_operations.load(Ordering::Relaxed),
            failed_operations: self.failed_operations.load(Ordering::Relaxed),
            success_rate: self.success_rate(),
            average_compute_time_ms: self.average_compute_time_ms(),
            operations_per_second: self.operations_per_second(),
            peak_memory_usage_mb: self.peak_memory_usage_mb.load(Ordering::Relaxed),
            average_queue_depth: self.average_queue_depth.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthLevel {
    Healthy,
    Warning,
    Critical,
    Failing,
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub health_level: HealthLevel,
    pub is_responsive: bool,
    pub success_rate: f64,
    pub average_latency_ms: u64,
    pub memory_pressure: f64,
    pub queue_health: QueueHealth,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepthTrend {
    Stable,
    Increasing,
    Decreasing,
    Volatile,
}

#[derive(Debug, Clone)]
pub struct QueueHealth {
    pub current_depth: usize,
    pub average_depth: usize,
    pub depth_trend: DepthTrend,
    pub oldest_sample_age: Duration,
    pub is_stable: bool,
}

#[derive(Debug)]
pub struct QueueDepthSample {
    pub timestamp: Instant,
    pub depth: usize,
    pub pending_operations: usize,
    pub memory_usage_mb: usize,
}

pub struct GpuHealthMonitor {
    metrics: GpuMetrics,
    pub queue_depth_tracker: VecDeque<QueueDepthSample>,
    performance_counter: PerformanceCounter,
    max_queue_samples: usize,
}

impl GpuHealthMonitor {
    pub fn new(max_queue_samples: usize) -> Self {
        Self {
            metrics: GpuMetrics::new(),
            queue_depth_tracker: VecDeque::with_capacity(max_queue_samples),
            performance_counter: PerformanceCounter::new(),
            max_queue_samples,
        }
    }

    /// Create a health monitor with default settings
    pub fn with_defaults() -> Self {
        return Self::new(300);
    }

    /// Record a successful operation
    pub fn record_operation_success(&self, duration: Duration, memory_usage_mb: usize, queue_depth: usize) {
        self.metrics.record_operation_completed(duration);
        self.performance_counter.record_success(duration);
        self.performance_counter.update_peak_memory(memory_usage_mb);
        self.performance_counter.update_queue_depth(queue_depth);
        self.update_metrics(memory_usage_mb, queue_depth);
    }

    /// Record a failed operation
    pub fn record_operation_failure(&self, duration: Duration, memory_usage_mb: usize, queue_depth: usize) {
        self.metrics.record_operation_failed(duration);
        self.performance_counter.record_failed(duration);
        self.performance_counter.update_peak_memory(memory_usage_mb);
        self.performance_counter.update_queue_depth(queue_depth);
        self.update_metrics(memory_usage_mb, queue_depth);
    }

    /// Update system metrics
    fn update_metrics(&self, memory_usage_mb: usize, queue_depth: usize) {
        self.metrics.update_memory_usage(memory_usage_mb);
        self.metrics.update_queue_depth(queue_depth);
    }

    /// Sample current queue depth and system state
    pub fn sample_queue_depth(&mut self, pending_operations: usize, memory_usage_mb: usize) {
        self.metrics.update_queue_depth(pending_operations);
        self.metrics.update_memory_usage(memory_usage_mb);

        let sample = QueueDepthSample {
            timestamp: Instant::now(),
            depth: pending_operations,
            pending_operations,
            memory_usage_mb,
        };

        self.queue_depth_tracker.push_back(sample);

        while self.max_queue_samples < self.queue_depth_tracker.len() {
            self.queue_depth_tracker.pop_front();
        }
    }

    /// Analyze queue depth trend over recent samples
    fn analyze_queue_trend(&self) -> DepthTrend {
        if self.queue_depth_tracker.len() < 10 {
            return DepthTrend::Stable;
        }

        let sample_count = self.queue_depth_tracker.len().min(30);
        let half_size = sample_count / 2;

        if half_size == 0 {
            return DepthTrend::Stable;
        }

        let recent_samples: Vec<_> = self.queue_depth_tracker.iter().rev().take(30).collect();
        let first_half: f64 = recent_samples.iter().rev().take(15).map(|s| s.depth as f64).sum::<f64>() / 15.0;
        let second_half: f64 = recent_samples.iter().take(15).map(|s| s.depth as f64).sum::<f64>() / 15.0;
        let variance: f64 = recent_samples
            .iter()
            .map(|s| {
                let diff = s.depth as f64 - ((first_half + second_half) / 2.0);
                diff * diff
            })
            .sum::<f64>()
            / (recent_samples.len() as f64);
        let std_dev = variance.sqrt();
        let mean = (first_half + second_half) / 2.0;
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        if 0.5 < coefficient_of_variation {
            return DepthTrend::Volatile;
        } else {
            let diff = second_half - first_half;

            if diff.abs() < 1.0 {
                return DepthTrend::Stable;
            } else if diff > 0.0 {
                return DepthTrend::Increasing;
            } else {
                return DepthTrend::Decreasing;
            }
        }
    }

    /// Get queue health assessment
    pub fn get_queue_health(&self) -> QueueHealth {
        let current_depth = self.metrics.queue_depth();
        let average_depth = self.performance_counter.average_queue_depth.load(Ordering::Relaxed);
        let depth_trend = self.analyze_queue_trend();
        let oldest_sample_age = self.queue_depth_tracker.front().map(|s| s.timestamp.elapsed()).unwrap_or(Duration::from_secs(0));
        let is_stable = matches!(depth_trend, DepthTrend::Stable | DepthTrend::Decreasing) && (current_depth < 50); // Arbitrary threshold

        return QueueHealth {
            current_depth,
            average_depth,
            depth_trend,
            oldest_sample_age,
            is_stable,
        };
    }

    /// Assess overall system health
    pub fn assess_health(&self) -> HealthStatus {
        let success_rate = self.metrics.success_rate();
        let is_responsive = self.metrics.is_responsive();
        let average_latency_ms = self.metrics.average_latency_ms();
        let memory_usage_mb = self.metrics.memory_usage_mb();
        let peak_memory_mb = self.performance_counter.peak_memory_usage_mb.load(Ordering::Relaxed);

        let memory_pressure = if peak_memory_mb > 0 { (memory_usage_mb as f64) / (peak_memory_mb as f64) } else { 0.0 };

        let queue_health = self.get_queue_health();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let health_level;

        if is_responsive == false {
            issues.push("System is not responsive (no heartbeat in >30s)".to_string());
            recommendations.push("Check for hanging operations or deadlocks".to_string());

            health_level = HealthLevel::Failing;
        } else if success_rate < 0.5 {
            issues.push(format!("Low success rate: {:.1}%", success_rate * 100.0));
            recommendations.push("Check circuit breaker status and error logs".to_string());

            health_level = HealthLevel::Critical;
        } else if success_rate < 0.8 || average_latency_ms > 5000 || memory_pressure > 0.9 {
            if success_rate < 0.8 {
                issues.push(format!("Moderate success rate: {:.1}%", success_rate * 100.0));
            }
            if average_latency_ms > 5000 {
                issues.push(format!("High latency: {}ms", average_latency_ms));
                recommendations.push("Check for resource contention or inefficient shaders".to_string());
            }
            if memory_pressure > 0.9 {
                issues.push(format!("High memory pressure: {:.1}%", memory_pressure * 100.0));
                recommendations.push("Consider enabling memory cleanup or reducing batch sizes".to_string());
            }

            health_level = HealthLevel::Critical;
        } else if success_rate < 0.95 || average_latency_ms > 1000 || memory_pressure > 0.7 {
            if success_rate < 0.95 {
                issues.push(format!("Success rate could be better: {:.1}%", success_rate * 100.0));
            }
            if average_latency_ms > 1000 {
                issues.push(format!("Elevated latency: {}ms", average_latency_ms));
            }
            if memory_pressure > 0.7 {
                issues.push(format!("Moderate memory pressure: {:.1}%", memory_pressure * 100.0));
            }

            health_level = HealthLevel::Warning;
        } else {
            health_level = HealthLevel::Healthy;
        };

        match queue_health.depth_trend {
            DepthTrend::Increasing => {
                issues.push("Queue depth is increasing".to_string());
                recommendations.push("Monitor for potential backlog or resource exhaustion".to_string());
            }
            DepthTrend::Volatile => {
                issues.push("Queue depth is volatile".to_string());
                recommendations.push("Check for bursty workload patterns or resource contention".to_string());
            }
            _ => {}
        }

        if queue_health.current_depth > 100 {
            issues.push(format!("High queue depth: {}", queue_health.current_depth));
            recommendations.push("Consider increasing concurrency limits or reducing task submission rate".to_string());
        }

        return HealthStatus {
            health_level,
            is_responsive,
            success_rate,
            average_latency_ms,
            memory_pressure,
            queue_health,
            issues,
            recommendations,
        };
    }

    /// Generate a comprehensive health report
    pub fn generate_health_report(&self) -> String {
        let health = self.assess_health();
        let perf_stats = self.performance_counter.get_stats();

        let mut report = String::new();

        report.push_str(&format!("=== GPU Health Monitor ===\n"));
        report.push_str(&format!("Overall Health: {:?}\n", health.health_level));
        report.push_str(&format!("Responsive: {}\n", health.is_responsive));
        report.push_str(&format!("Success Rate: {:.2}%\n", health.success_rate * 100.0));
        report.push_str(&format!("Average Latency: {}ms\n", health.average_latency_ms));
        report.push_str(&format!("Memory Pressure: {:.1}%\n", health.memory_pressure * 100.0));
        report.push_str(&format!("\n=== Performance Statistics ===\n"));
        report.push_str(&format!("Total Operations: {}\n", perf_stats.total_operations));
        report.push_str(&format!("Operations/sec: {:.2}\n", perf_stats.operations_per_second));
        report.push_str(&format!("Peak Memory: {}MB\n", perf_stats.peak_memory_usage_mb));
        report.push_str(&format!("\n=== Queue Health ===\n"));
        report.push_str(&format!("Current Depth: {}\n", health.queue_health.current_depth));
        report.push_str(&format!("Average Depth: {}\n", health.queue_health.average_depth));
        report.push_str(&format!("Trend: {:?}\n", health.queue_health.depth_trend));
        report.push_str(&format!("Stable: {}\n", health.queue_health.is_stable));

        if !health.issues.is_empty() {
            report.push_str(&format!("\n=== Issues ===\n"));
            for issue in &health.issues {
                report.push_str(&format!("• {}\n", issue));
            }
        }

        if !health.recommendations.is_empty() {
            report.push_str(&format!("\n=== Recommendations ===\n"));
            for rec in &health.recommendations {
                report.push_str(&format!("• {}\n", rec));
            }
        }

        return report;
    }

    /// Log health status at appropriate log level
    pub fn log_health_status(&self) {
        let health = self.assess_health();

        match health.health_level {
            HealthLevel::Healthy => {
                debug!(
                    success_rate = health.success_rate,
                    latency_ms = health.average_latency_ms,
                    queue_depth = health.queue_health.current_depth,
                    "GPU system healthy"
                );
            }
            HealthLevel::Warning => {
                warn!(
                    success_rate = health.success_rate,
                    latency_ms = health.average_latency_ms,
                    memory_pressure = health.memory_pressure,
                    issues = health.issues.len(),
                    "GPU system health warning"
                );
            }
            HealthLevel::Critical => {
                error!(
                    success_rate = health.success_rate,
                    latency_ms = health.average_latency_ms,
                    memory_pressure = health.memory_pressure,
                    responsive = health.is_responsive,
                    issues = ?health.issues,
                    "GPU system health critical"
                );
            }
            HealthLevel::Failing => {
                error!(
                    responsive = health.is_responsive,
                    heartbeat_age = self.metrics.seconds_since_heartbeat(),
                    issues = ?health.issues,
                    "GPU system failing"
                );
            }
        }
    }

    /// Check if emergency shutdown should be triggered
    pub fn should_emergency_shutdown(&self) -> bool {
        let health = self.assess_health();

        match health.health_level {
            HealthLevel::Failing => {
                return true;
            }
            HealthLevel::Critical => {
                return (health.is_responsive == false) || (health.success_rate < 0.1) || (500 < health.queue_health.current_depth);
            }
            _ => {
                return false;
            }
        }
    }

    /// Get access to underlying metrics for external monitoring systems
    pub fn get_metrics(&self) -> &GpuMetrics {
        return &self.metrics;
    }

    /// Get access to performance counter for detailed statistics
    pub fn get_performance_counter(&self) -> &PerformanceCounter {
        return &self.performance_counter;
    }

    /// Get recent queue depth samples for analysis
    pub fn get_recent_queue_samples(&self, count: usize) -> Vec<&QueueDepthSample> {
        return self.queue_depth_tracker.iter().rev().take(count).collect();
    }
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

#[derive(Debug)]
pub struct GradientEntry {
    pub gradient: Tensor,
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
