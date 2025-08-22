use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use tokio::sync::Semaphore;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};
use wgpu::{
    Adapter, AdapterInfo, BufferUsages, CommandBuffer, CommandEncoder, CommandEncoderDescriptor,
    Device, DeviceType, Limits, Queue,
};

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
    Tensor(Tensor),
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
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        batch_size: usize,
        max_active_buffers: usize,
        default_timeout: Duration,
    ) -> Self {
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
    pub fn get_encoder(
        &mut self,
        label: Option<&str>,
        _resource_tracker: &ResourceTracker,
    ) -> Result<CommandEncoder, GpuError> {
        if let Some(encoder) = self.available_encoders.pop_front() {
            return Ok(encoder);
        } else {
            let encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label });
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
            tracing::debug!(
                cleaned_buffers = cleaned_count,
                remaining_buffers = self.active_buffers.len(),
                "Cleaned up timed out command buffers"
            );
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

        let buffers: Vec<CommandBuffer> = self
            .active_buffers
            .drain(..)
            .map(|active| active.buffer)
            .collect();

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
    pub fn submit_buffer(
        &mut self,
        buffer: CommandBuffer,
        operation_id: OperationId,
        timeout: Option<Duration>,
        _guard: CommandBufferGuard<'_>,
    ) -> Result<(), GpuError> {
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
        return self
            .active_buffers
            .iter()
            .map(|buffer| {
                (
                    buffer.operation_id,
                    buffer.created_at.elapsed(),
                    buffer.get_remaining_time(),
                )
            })
            .collect();
    }

    /// Emergency cleanup - drop all pending buffers
    pub fn emergency_cleanup(&mut self) -> usize {
        let count = self.active_buffers.len();

        if 0 < count {
            tracing::error!(
                dropped_buffers = count,
                "Emergency cleanup - dropping all pending command buffers"
            );

            self.active_buffers.clear();
            self.available_encoders.clear();
        }

        count
    }

    /// Get current pool statistics
    pub fn get_status(&self) -> CommandBufferStatus {
        let oldest_buffer_age = self
            .active_buffers
            .first()
            .map(|buffer| buffer.created_at.elapsed());

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
    pub fn new(
        pool: &'a mut CommandBufferPool,
        operation_id: OperationId,
        label: Option<&str>,
        resource_tracker: &ResourceTracker,
    ) -> Result<Self, GpuError> {
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

    pub fn finish_and_submit(
        mut self,
        timeout: Option<Duration>,
        guard: CommandBufferGuard<'a>,
    ) -> Result<(), GpuError> {
        let encoder = self.encoder.take().expect("Encoder already consumed");
        let buffer = encoder.finish();

        self.pool
            .submit_buffer(buffer, self.operation_id, timeout, guard)
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
    pub fn success(
        operation_id: OperationId,
        variable_id: VariableId,
        duration: Duration,
        gradient_norm: Option<f32>,
    ) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::Success,
            duration,
            gradient_norm,
        };
    }

    /// Create a failed task completion
    pub fn failed(
        operation_id: OperationId,
        variable_id: VariableId,
        duration: Duration,
        error: String,
    ) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::Failed(error),
            duration,
            gradient_norm: None,
        };
    }

    /// Create a timeout task completion
    pub fn timed_out(
        operation_id: OperationId,
        variable_id: VariableId,
        duration: Duration,
    ) -> Self {
        return Self {
            operation_id,
            variable_id,
            status: TaskStatus::TimedOut,
            duration,
            gradient_norm: None,
        };
    }

    /// Create a cancelled task completion
    pub fn cancelled(
        operation_id: OperationId,
        variable_id: VariableId,
        duration: Duration,
    ) -> Self {
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

    pub fn from_timed_result<T, E>(
        operation_id: OperationId,
        variable_id: VariableId,
        start_time: Instant,
        result: Result<T, E>,
        gradient_norm: Option<f32>,
    ) -> Self
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.last_state_change.store(now, Ordering::Relaxed);
    }

    fn open(&self) {
        self.state
            .store(CircuitBreakerState::Open as u8, Ordering::Release);
        self.update_state_change_time();
    }

    fn open_half(&self) {
        self.state
            .store(CircuitBreakerState::HalfOpen as u8, Ordering::Release);
        self.update_state_change_time();

        tracing::info!("Circuit breaker transitioning to HALF_OPEN");
    }

    fn close(&self) {
        self.state
            .store(CircuitBreakerState::Closed as u8, Ordering::Release);
        self.failure_count.store(0, Ordering::Relaxed);
        self.update_state_change_time();
    }

    pub fn force_open(&self) {
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

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
            GradientTask::Add {
                target_variable, ..
            } => *target_variable,
            GradientTask::Mul {
                target_variable, ..
            } => *target_variable,
            GradientTask::MatMul {
                target_variable, ..
            } => *target_variable,
            GradientTask::Transpose {
                target_variable, ..
            } => *target_variable,
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
            GradientTask::MatMul {
                output_grad,
                other_input,
                ..
            } => {
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
        let failure_rate = if self.total_submitted > 0 {
            self.total_failed as f64 / self.total_submitted as f64
        } else {
            0.0
        };

        return self.is_shutdown
            || self.circuit_breaker_stats.indicates_problems()
            || failure_rate > 0.1
            || (self.queue_length > 1000 && self.active_tasks == self.max_concurrent_tasks);
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
    task_queue: flume::Receiver<GradientTask>,
    task_sender: flume::Sender<GradientTask>,
    active_tasks: Arc<AtomicUsize>,
    max_concurrent_tasks: usize,
    completion_notifier: watch::Sender<TaskCompletion>,
    circuit_breaker: Arc<CircuitBreaker>,

    semaphore: Arc<Semaphore>,
    shutdown_signal: Arc<AtomicBool>,
    task_timeout: Duration,

    total_tasks_submitted: AtomicUsize,
    total_tasks_completed: AtomicUsize,
    total_tasks_failed: AtomicUsize,
}

impl GradientTaskManager {
    /// Create a new gradient task manager
    pub fn new(
        max_concurrent_tasks: usize,
        circuit_breaker: CircuitBreaker,
        task_timeout: Duration,
    ) -> (Self, watch::Receiver<TaskCompletion>) {
        let (task_sender, task_queue) = flume::unbounded();
        let (completion_notifier, completion_receiver) = watch::channel(TaskCompletion::success(
            OperationId(0),
            VariableId(0),
            Duration::from_millis(0),
            None,
        ));

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
    pub fn for_gpu_gradients(
        max_concurrent_tasks: usize,
    ) -> (Self, watch::Receiver<TaskCompletion>) {
        Self::new(
            max_concurrent_tasks,
            CircuitBreaker::for_gpu_operations(),
            Duration::from_secs(10),
        )
    }

    /// Submit a gradient task for execution
    pub async fn submit_task(&self, task: GradientTask) -> Result<(), String> {
        if self.shutdown_signal.load(Ordering::Acquire) == true {
            return Err("Task manager is shutting down".to_string());
        }

        if self.circuit_breaker.is_request_allowed() == false {
            let completion = TaskCompletion::failed(
                task.get_operation_id(),
                task.get_target_variable(),
                Duration::from_millis(0),
                "Circuit breaker is open".to_string(),
            );

            let _ = self.completion_notifier.send(completion);
            return Err("Circuit breaker is open, rejecting task".to_string());
        }

        self.total_tasks_submitted.fetch_add(1, Ordering::Relaxed);
        self.task_sender
            .send_async(task)
            .await
            .map_err(|_| "Failed to queue task, manager may be shutting down".to_string())?;

        return Ok(());
    }

    pub async fn run(&self, resource_tracker: Arc<ResourceTracker>) {
        info!(
            "Starting gradient task manager with {} max concurrent tasks",
            self.max_concurrent_tasks
        );

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

                let _guard = TaskGuard {
                    active_tasks: active_tasks.clone(),
                };
                let start_time = Instant::now();
                let result = timeout(
                    task_timeout,
                    Self::execute_gradient_task(task.clone(), resource_tracker, shutdown_signal),
                )
                .await;

                let completion = match result {
                    Ok(Ok(gradient_norm)) => {
                        circuit_breaker.record_success();
                        TaskCompletion::success(
                            task.get_operation_id(),
                            task.get_target_variable(),
                            start_time.elapsed(),
                            gradient_norm,
                        )
                    }
                    Ok(Err(error)) => {
                        circuit_breaker.record_failure();
                        TaskCompletion::failed(
                            task.get_operation_id(),
                            task.get_target_variable(),
                            start_time.elapsed(),
                            error,
                        )
                    }
                    Err(_) => {
                        circuit_breaker.record_failure();
                        TaskCompletion::timed_out(
                            task.get_operation_id(),
                            task.get_target_variable(),
                            start_time.elapsed(),
                        )
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
    async fn execute_gradient_task(
        task: GradientTask,
        resource_tracker: Arc<ResourceTracker>,
        shutdown_signal: Arc<AtomicBool>,
    ) -> Result<Option<f32>, String> {
        if shutdown_signal.load(Ordering::Acquire) == true {
            return Err("Shutdown requested".to_string());
        }

        let _operation_guard = resource_tracker
            .reserve_operation()
            .map_err(|e| format!("Failed to reserve GPU resources: {}", e))?;

        match task {
            GradientTask::Add {
                output_grad,
                target_variable,
                ..
            } => {
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
            GradientTask::Transpose {
                output_grad,
                target_variable,
                ..
            } => {
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
        TaskManagerStats {
            active_tasks: self.active_tasks.load(Ordering::Acquire),
            max_concurrent_tasks: self.max_concurrent_tasks,
            total_submitted: self.total_tasks_submitted.load(Ordering::Acquire),
            total_completed: self.total_tasks_completed.load(Ordering::Acquire),
            total_failed: self.total_tasks_failed.load(Ordering::Acquire),
            queue_length: self.task_queue.len(),
            available_permits: self.semaphore.available_permits(),
            circuit_breaker_stats: self.circuit_breaker.get_stats(),
            is_shutdown: self.shutdown_signal.load(Ordering::Acquire),
        }
    }

    /// Check if the task manager is healthy
    pub fn is_healthy(&self) -> bool {
        return (self.shutdown_signal.load(Ordering::Acquire) == false)
            && (self.circuit_breaker.is_healthy() == true)
            && (self.active_tasks.load(Ordering::Acquire) < self.max_concurrent_tasks);
    }
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
