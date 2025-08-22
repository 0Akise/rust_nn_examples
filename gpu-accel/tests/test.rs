#[cfg(test)]
mod tests {
    use gpu_accel::migration::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use tokio::time::sleep;
    use wgpu::{
        Device, DeviceDescriptor, Features, Instance, Limits, Queue, RequestAdapterOptions,
    };

    async fn create_test_gpu_context() -> (Device, Queue) {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            })
            .await
            .expect("Failed to create adapter for testing");

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("Test Device"),
                required_features: Features::empty(),
                required_limits: Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device for testing");

        (device, queue)
    }

    fn next_operation_id() -> OperationId {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        OperationId(COUNTER.fetch_add(1, Ordering::Relaxed) as u64)
    }

    #[test]
    fn test_resource_limits_creation() {
        let generic = ResourceLimits::generic();
        assert_eq!(generic.max_concurrent_operations, 8);
        assert_eq!(generic.max_command_buffers, 16);
        assert_eq!(generic.queue_depth_limit, 32);

        let limits = Limits::downlevel_defaults();
        let estimated_mb = ResourceLimits::estimate_memory(&limits);
        assert!(estimated_mb > 0, "Should estimate some memory");
    }

    #[test]
    fn test_resource_tracker_creation() {
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let usage = tracker.current_usage();
        assert_eq!(usage.gpu_operations, 0);
        assert_eq!(usage.command_buffers, 0);
        assert_eq!(usage.memory_usage_mb, 0);
        assert_eq!(usage.total_queue_depth, 0);
    }

    #[test]
    fn test_resource_tracker_basic_reservations() {
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let queue_guard = tracker
            .reserve_queue_slot()
            .expect("Should reserve queue slot");
        assert_eq!(tracker.current_usage().total_queue_depth, 1);

        drop(queue_guard);
        assert_eq!(tracker.current_usage().total_queue_depth, 0);

        let cmd_guard = tracker
            .reserve_command_buffer()
            .expect("Should reserve command buffer");
        assert_eq!(tracker.current_usage().command_buffers, 1);

        drop(cmd_guard);
        assert_eq!(tracker.current_usage().command_buffers, 0);

        let mem_guard = tracker.reserve_memory(100).expect("Should reserve memory");
        assert_eq!(tracker.current_usage().memory_usage_mb, 100);

        drop(mem_guard);
        assert_eq!(tracker.current_usage().memory_usage_mb, 0);
    }

    #[test]
    fn test_resource_tracker_limits_enforcement() {
        let mut limits = ResourceLimits::generic();
        limits.max_concurrent_operations = 2;
        limits.queue_depth_limit = 3;
        limits.max_command_buffers = 2;
        limits.max_memory_usage_mb = 100;

        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let _guard1 = tracker
            .reserve_queue_slot()
            .expect("First reservation should work");
        let _guard2 = tracker
            .reserve_queue_slot()
            .expect("Second reservation should work");
        let _guard3 = tracker
            .reserve_queue_slot()
            .expect("Third reservation should work");

        let result = tracker.reserve_queue_slot();
        assert!(matches!(
            result,
            Err(GpuError::QueueOverflow {
                active: 3,
                limit: 3
            })
        ));

        let _cmd1 = tracker
            .reserve_command_buffer()
            .expect("First cmd buffer should work");
        let _cmd2 = tracker
            .reserve_command_buffer()
            .expect("Second cmd buffer should work");

        let result = tracker.reserve_command_buffer();
        assert!(matches!(
            result,
            Err(GpuError::QueueOverflow {
                active: 2,
                limit: 2
            })
        ));

        let _mem1 = tracker
            .reserve_memory(50)
            .expect("First memory reservation should work");
        let _mem2 = tracker
            .reserve_memory(30)
            .expect("Second memory reservation should work");

        let result = tracker.reserve_memory(30);
        assert!(matches!(
            result,
            Err(GpuError::MemoryExhaustion {
                requested: 30,
                available: 20
            })
        ));
    }

    #[test]
    fn test_resource_tracker_emergency_stop() {
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let _guard = tracker.reserve_queue_slot().expect("Should work normally");

        tracker.emergency_stop_trigger();

        let result = tracker.reserve_queue_slot();
        assert!(matches!(
            result,
            Err(GpuError::CircuitBreakerOpen { failures: 0 })
        ));

        tracker.emergency_stop_reset();
        let _guard2 = tracker
            .reserve_queue_slot()
            .expect("Should work after reset");
    }

    #[test]
    fn test_resource_tracker_concurrent_access() {
        let limits = ResourceLimits::generic();
        let tracker = Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));
        let success_count = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..20)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                let success_count = Arc::clone(&success_count);
                let failure_count = Arc::clone(&failure_count);

                thread::spawn(move || {
                    if let Ok(_guard) = tracker.reserve_queue_slot() {
                        success_count.fetch_add(1, Ordering::Relaxed);
                        thread::sleep(Duration::from_millis(10));
                    } else {
                        failure_count.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let successes = success_count.load(Ordering::Relaxed);
        let failures = failure_count.load(Ordering::Relaxed);

        println!(
            "Concurrent test: {} successes, {} failures",
            successes, failures
        );
        assert_eq!(successes + failures, 20);
        assert!(successes > 0, "Should have some successes");
    }

    #[tokio::test]
    async fn test_command_buffer_pool_creation() {
        let (device, queue) = create_test_gpu_context().await;

        let pool = CommandBufferPool::new(device, queue, 4, 8, Duration::from_secs(5));

        let status = pool.get_status();
        assert_eq!(status.active_buffers, 0);
        assert_eq!(status.available_encoders, 0);
        assert_eq!(status.total_submissions, 0);
        assert_eq!(status.timeout_count, 0);
    }

    #[tokio::test]
    async fn test_command_buffer_pool_encoder_management() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let mut pool = CommandBufferPool::new(device, queue, 4, 8, Duration::from_secs(5));

        let encoder = pool
            .get_encoder(Some("Test Encoder"), &tracker)
            .expect("Should get encoder");

        pool.return_encoder(encoder);

        let status = pool.get_status();
        assert_eq!(status.available_encoders, 1);

        let _encoder2 = pool
            .get_encoder(Some("Test Encoder 2"), &tracker)
            .expect("Should reuse encoder");

        let status = pool.get_status();
        assert_eq!(status.available_encoders, 0);
    }

    #[tokio::test]
    async fn test_command_buffer_pool_batch_submission() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let mut pool = CommandBufferPool::new(device, queue, 2, 8, Duration::from_secs(5));

        for i in 0..3 {
            let encoder = pool
                .get_encoder(Some(&format!("Test {}", i)), &tracker)
                .expect("Should get encoder");
            let buffer = encoder.finish();
            let guard = tracker.reserve_command_buffer().expect("Should get guard");

            pool.submit_buffer(buffer, next_operation_id(), None, guard)
                .expect("Should submit buffer");
        }

        let status = pool.get_status();
        assert!(
            status.total_submissions > 0,
            "Should have submitted batches"
        );
        println!("Total submissions: {}", status.total_submissions);
    }

    #[tokio::test]
    async fn test_command_buffer_pool_timeout_handling() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let mut pool = CommandBufferPool::new(device, queue, 10, 8, Duration::from_millis(50));

        let encoder = pool
            .get_encoder(Some("Timeout Test"), &tracker)
            .expect("Should get encoder");
        let buffer = encoder.finish();
        let guard = tracker.reserve_command_buffer().expect("Should get guard");

        pool.submit_buffer(
            buffer,
            next_operation_id(),
            Some(Duration::from_millis(10)),
            guard,
        )
        .expect("Should submit buffer");

        sleep(Duration::from_millis(100)).await;

        let submitted = pool.submit_batch().expect("Should handle cleanup");

        let status = pool.get_status();
        println!(
            "After timeout - active: {}, timeouts: {}, submitted: {}",
            status.active_buffers, status.timeout_count, submitted
        );

        assert_eq!(status.active_buffers, 0);
    }

    #[tokio::test]
    async fn test_managed_encoder_raii() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");
        let mut pool = CommandBufferPool::new(device, queue, 4, 8, Duration::from_secs(5));
        let operation_id = next_operation_id();
        let initial_status = pool.get_status();
        assert_eq!(initial_status.available_encoders, 0);

        {
            let _managed_encoder =
                ManagedEncoder::new(&mut pool, operation_id, Some("RAII Test"), &tracker)
                    .expect("Should create managed encoder");
        }

        let final_status = pool.get_status();
        assert_eq!(final_status.available_encoders, 1);
    }

    #[tokio::test]
    async fn test_managed_encoder_explicit_submission() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let mut pool = CommandBufferPool::new(device, queue, 4, 8, Duration::from_secs(5));
        let operation_id = next_operation_id();

        let managed_encoder =
            ManagedEncoder::new(&mut pool, operation_id, Some("Submission Test"), &tracker)
                .expect("Should create managed encoder");

        let guard = tracker.reserve_command_buffer().expect("Should get guard");

        managed_encoder
            .finish_and_submit(None, guard)
            .expect("Should submit successfully");

        let status = pool.get_status();
        assert!(status.total_submissions > 0 || status.active_buffers > 0);
    }

    #[tokio::test]
    async fn test_command_buffer_pool_health_check() {
        let (device, queue) = create_test_gpu_context().await;
        let pool = CommandBufferPool::new(device, queue, 4, 8, Duration::from_secs(5));

        assert!(pool.health_check(), "Pool should start healthy");

        pool.total_submissions.store(100, Ordering::Relaxed);
        pool.timeout_count.store(5, Ordering::Relaxed);

        assert!(
            pool.health_check(),
            "Pool should be healthy with low timeout rate"
        );

        pool.timeout_count.store(15, Ordering::Relaxed);

        assert!(
            !pool.health_check(),
            "Pool should be unhealthy with high timeout rate"
        );
    }

    #[tokio::test]
    async fn test_command_buffer_pool_emergency_cleanup() {
        let (device, queue) = create_test_gpu_context().await;
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let mut pool = CommandBufferPool::new(device, queue, 10, 8, Duration::from_secs(5));

        for i in 0..3 {
            let encoder = pool
                .get_encoder(Some(&format!("Emergency {}", i)), &tracker)
                .expect("Should get encoder");
            let buffer = encoder.finish();
            let guard = tracker.reserve_command_buffer().expect("Should get guard");

            pool.submit_buffer(buffer, next_operation_id(), None, guard)
                .expect("Should submit buffer");
        }

        let status_before = pool.get_status();
        assert!(
            status_before.active_buffers > 0,
            "Should have active buffers"
        );

        let cleaned = pool.emergency_cleanup();
        assert_eq!(cleaned, status_before.active_buffers);

        let status_after = pool.get_status();
        assert_eq!(status_after.active_buffers, 0);
        assert_eq!(status_after.available_encoders, 0);
    }

    #[test]
    fn test_operation_guard_split() {
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let operation_guard = tracker
            .reserve_operation()
            .expect("Should reserve operation");

        let usage = tracker.current_usage();
        assert_eq!(usage.total_queue_depth, 1);
        assert_eq!(usage.gpu_operations, 1);

        let (queue_guard, gpu_guard) = operation_guard.split();

        let usage = tracker.current_usage();
        assert_eq!(usage.total_queue_depth, 1);
        assert_eq!(usage.gpu_operations, 1);

        drop(queue_guard);
        let usage = tracker.current_usage();
        assert_eq!(usage.total_queue_depth, 0);
        assert_eq!(usage.gpu_operations, 1);

        drop(gpu_guard);
        let usage = tracker.current_usage();
        assert_eq!(usage.total_queue_depth, 0);
        assert_eq!(usage.gpu_operations, 0);
    }

    #[test]
    fn test_resource_limits_autotune() {
        let _ = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        println!("Autotune test requires real GPU adapter - skipping detailed verification");

        let generic_limits = ResourceLimits::generic();
        assert!(generic_limits.max_concurrent_operations > 0);
        assert!(generic_limits.max_command_buffers > 0);
        assert!(generic_limits.max_memory_usage_mb > 0);
    }

    #[tokio::test]
    async fn test_resource_tracker_performance() {
        let limits = ResourceLimits::generic();
        let tracker = Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _guard = tracker.reserve_queue_slot().expect("Should reserve");
        }

        let duration = start.elapsed();
        println!(
            "Reservation performance: {} ops in {:?} ({:.2} ops/ms)",
            iterations,
            duration,
            iterations as f64 / duration.as_millis() as f64
        );

        assert!(
            duration < Duration::from_millis(100),
            "Reservations should be fast"
        );
    }
}
