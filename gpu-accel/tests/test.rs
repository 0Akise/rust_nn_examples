#[cfg(test)]
mod tests {
    use gpu_accel::migration::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use tokio::time::sleep;
    use wgpu::{
        BufferUsages, Device, DeviceDescriptor, Features, Instance, Limits, Queue,
        RequestAdapterOptions,
    };

    async fn create_test_gpu_context() -> (Device, Queue) {
        let context_creation = async {
            let instance = Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to create adapter for testing");

            let info = adapter.get_info();

            if info.device_type == wgpu::DeviceType::Cpu {
                panic!("Test failed to obtain GPU adapter, got CPU fallback instead. Check your GPU drivers and WGPU installation.");
            }

            println!(
                "Test using GPU: {} (Vendor: {:04X}, Type: {:?}, Backend: {:?})",
                info.name, info.vendor, info.device_type, info.backend
            );

            let (device, queue) = adapter
                .request_device(&DeviceDescriptor {
                    label: Some("Test Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("Failed to create device for testing");

            (device, queue)
        };

        tokio::time::timeout(Duration::from_secs(10), context_creation)
            .await
            .expect("GPU context creation should not hang")
    }

    fn next_operation_id() -> OperationId {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        OperationId(COUNTER.fetch_add(1, Ordering::Relaxed) as u64)
    }

    fn next_variable_id() -> VariableId {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        VariableId(COUNTER.fetch_add(1, Ordering::Relaxed) as u64)
    }

    fn create_test_tensor(dims: Vec<usize>) -> Tensor {
        let total_elements: usize = dims.iter().product();
        let data: Vec<f32> = (0..total_elements).map(|i| i as f32).collect();
        let shape = Shape { dims };
        Tensor {
            data: Arc::new(data),
            shape,
        }
    }

    #[tokio::test]
    async fn test_gradient_task_manager_creation() {
        let (task_manager, completion_receiver) = GradientTaskManager::for_gpu_gradients(4);

        let stats = task_manager.get_stats();
        assert_eq!(stats.max_concurrent_tasks, 4);
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.total_submitted, 0);
        assert!(stats.is_shutdown == false);
        assert!(task_manager.is_healthy());
    }

    #[tokio::test]
    async fn test_gradient_task_submission() {
        let test_timeout = Duration::from_secs(10);
        let result = tokio::time::timeout(test_timeout, async {
            let (task_manager, mut completion_receiver) = GradientTaskManager::for_gpu_gradients(2);
            let limits = ResourceLimits::generic();
            let resource_tracker =
                Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));

            let task_manager_arc = Arc::new(task_manager);
            let manager_clone = task_manager_arc.clone();
            let tracker_clone = resource_tracker.clone();

            let manager_handle = tokio::spawn(async move {
                manager_clone.run(tracker_clone).await;
            });

            for i in 0..2 {
                let gradient_tensor = create_test_tensor(vec![2, 2]);
                let task = GradientTask::Add {
                    output_grad: gradient_tensor,
                    target_variable: next_variable_id(),
                    operation_id: next_operation_id(),
                };

                let result = task_manager_arc.submit_task(task).await;
                assert!(result.is_ok(), "Task submission {} should succeed", i);
            }

            let mut completions = 0;
            let completion_timeout = Duration::from_secs(3);
            let start = Instant::now();

            while completions < 2 && start.elapsed() < completion_timeout {
                match tokio::time::timeout(
                    Duration::from_millis(500),
                    completion_receiver.changed(),
                )
                .await
                {
                    Ok(Ok(())) => {
                        let completion = completion_receiver.borrow_and_update();
                        if completion.is_success() {
                            completions += 1;
                            println!(
                                "Received completion {} for operation {:?}",
                                completions, completion.operation_id
                            );
                        }
                    }
                    Ok(Err(_)) => break,
                    Err(_) => continue,
                }
            }

            let stats = task_manager_arc.get_stats();
            println!(
                "Final stats: submitted={}, completed={}, active={}",
                stats.total_submitted, completions, stats.active_tasks
            );

            tokio::select! {
                _ = task_manager_arc.shutdown() => {},
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    println!("Shutdown timeout, forcing emergency shutdown");
                    task_manager_arc.emergency_shutdown();
                }
            }

            manager_handle.abort();

            assert!(
                stats.total_submitted >= 2,
                "Should have submitted at least 2 tasks"
            );
        })
        .await;

        assert!(result.is_ok(), "Test should complete within timeout");
    }

    #[tokio::test]
    async fn test_circuit_breaker_functionality() {
        let circuit_breaker = CircuitBreaker::for_gpu_operations();

        assert_eq!(circuit_breaker.get_state(), CircuitBreakerState::Closed);
        assert!(circuit_breaker.is_healthy());

        for _ in 0..3 {
            let allowed = circuit_breaker.is_request_allowed();
            assert!(allowed, "Request should be allowed in closed state");
            circuit_breaker.record_success();
        }

        let stats = circuit_breaker.get_stats();

        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.total_requests, 3);
        assert!((stats.success_rate - 1.0).abs() < 0.01);

        for i in 0..5 {
            let allowed = circuit_breaker.is_request_allowed();
            println!(
                "Failure {}: request allowed = {}, state = {:?}",
                i,
                allowed,
                circuit_breaker.get_state()
            );
            circuit_breaker.record_failure();
        }

        assert_eq!(circuit_breaker.get_state(), CircuitBreakerState::Open);
        assert!(!circuit_breaker.is_healthy());

        assert!(!circuit_breaker.is_request_allowed());
        assert!(!circuit_breaker.is_request_allowed());

        let final_stats = circuit_breaker.get_stats();
        assert_eq!(final_stats.failure_count, 5);
        assert!(final_stats.rejected_requests > 0);
    }

    #[tokio::test]
    async fn test_gpu_memory_manager_basic_allocation() {
        let test_timeout = Duration::from_secs(15);
        let result = tokio::time::timeout(test_timeout, async {
            let (device, queue) = create_test_gpu_context().await;
            let limits = ResourceLimits::generic();
            let resource_tracker =
                Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));
            let memory_manager = GpuMemoryManager::new(
                Arc::new(device),
                Arc::new(queue),
                resource_tracker,
                AllocationStrategy::Pooled,
                100,
            );
            let buffer_id = memory_manager
                .allocate_buffer(
                    1024,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    Some("Test Buffer"),
                    None,
                )
                .await
                .expect("Should allocate buffer");
            let buffer_ref = memory_manager.get_buffer(buffer_id);

            assert!(buffer_ref.is_some(), "Should retrieve allocated buffer");

            let stats = memory_manager.get_stats();

            println!(
                "Memory debug: active={}, pooled={}, efficiency={:.3}",
                stats.active_buffers,
                stats.pooled_buffers,
                stats.allocation_efficiency()
            );
            assert_eq!(stats.active_buffers, 1);
            assert!(stats.total_allocated_bytes >= 1024);
            assert!(memory_manager.is_healthy());

            memory_manager
                .release_buffer(buffer_id)
                .await
                .expect("Should release buffer");

            println!(
                "Memory stats after allocation: active_buffers={}, total_bytes={}",
                stats.active_buffers, stats.total_allocated_bytes
            );

            memory_manager.shutdown().await;
        })
        .await;

        assert!(
            result.is_ok(),
            "GPU memory manager test should complete within timeout"
        );
    }

    #[tokio::test]
    async fn test_gpu_memory_manager_pooling() {
        let test_timeout = Duration::from_secs(15);
        let result = tokio::time::timeout(test_timeout, async {
            let (device, queue) = create_test_gpu_context().await;
            let limits = ResourceLimits::generic();
            let resource_tracker =
                Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));

            let memory_manager = GpuMemoryManager::new(
                Arc::new(device),
                Arc::new(queue),
                resource_tracker,
                AllocationStrategy::Pooled,
                100,
            );

            let mut buffer_ids = Vec::new();
            for i in 0..3 {
                let buffer_id = memory_manager
                    .allocate_buffer(
                        4096,
                        BufferUsages::STORAGE,
                        Some(&format!("Pool Test Buffer {}", i)),
                        Some(AllocationStrategy::Pooled),
                    )
                    .await
                    .expect("Should allocate pooled buffer");

                buffer_ids.push(buffer_id);
            }

            for buffer_id in &buffer_ids[0..2] {
                memory_manager
                    .release_buffer(*buffer_id)
                    .await
                    .expect("Should release buffer");
            }

            tokio::time::sleep(Duration::from_millis(100)).await;

            let reused_buffer_id = memory_manager
                .allocate_buffer(
                    4096,
                    BufferUsages::STORAGE,
                    Some("Reused Buffer"),
                    Some(AllocationStrategy::Pooled),
                )
                .await
                .expect("Should allocate reused buffer");

            let stats = memory_manager.get_stats();
            println!(
                "Pooling test stats: active={}, pooled={}, efficiency={:.2}",
                stats.active_buffers,
                stats.pooled_buffers,
                stats.allocation_efficiency()
            );

            assert!(stats.active_buffers > 0);

            memory_manager.shutdown().await;
        })
        .await;

        assert!(
            result.is_ok(),
            "GPU memory pooling test should complete within timeout"
        );
    }

    #[tokio::test]
    async fn test_allocation_strategy_selection() {
        let small_pooled =
            AllocationStrategy::choose_for_usage(512 * 1024, BufferUsages::STORAGE, false);
        assert!(matches!(small_pooled, AllocationStrategy::Pooled));

        let large_direct =
            AllocationStrategy::choose_for_usage(20 * 1024 * 1024, BufferUsages::STORAGE, true);
        assert!(matches!(large_direct, AllocationStrategy::Direct));

        let streaming = AllocationStrategy::choose_for_usage(
            100 * 1024 * 1024,
            BufferUsages::MAP_READ | BufferUsages::MAP_WRITE,
            false,
        );
        assert!(matches!(streaming, AllocationStrategy::Streaming));

        assert!(AllocationStrategy::Pooled.supports_reuse());
        assert!(!AllocationStrategy::Direct.supports_reuse());
        assert!(
            AllocationStrategy::Direct.cleanup_priority()
                > AllocationStrategy::Pooled.cleanup_priority()
        );
    }

    #[tokio::test]
    async fn test_memory_exhaustion_handling() {
        let test_timeout = Duration::from_secs(10);
        let result = tokio::time::timeout(test_timeout, async {
            let (device, queue) = create_test_gpu_context().await;
            let limits = ResourceLimits::generic();
            let resource_tracker =
                Arc::new(ResourceTracker::new(limits).expect("Should create tracker"));

            let memory_manager = GpuMemoryManager::new(
                Arc::new(device),
                Arc::new(queue),
                resource_tracker,
                AllocationStrategy::Direct,
                1,
            );

            let result = memory_manager
                .allocate_buffer(
                    2 * 1024 * 1024,
                    BufferUsages::STORAGE,
                    Some("Too Large Buffer"),
                    None,
                )
                .await;

            assert!(result.is_err(), "Should fail with memory exhaustion");

            if let Err(GpuError::MemoryExhaustion {
                requested,
                available,
            }) = result
            {
                assert_eq!(requested, 2);
                assert!(available < 2);
                println!(
                    "Memory exhaustion correctly detected: requested={}MB, available={}MB",
                    requested, available
                );
            } else {
                panic!("Expected MemoryExhaustion error");
            }

            memory_manager.shutdown().await;
        })
        .await;

        assert!(
            result.is_ok(),
            "Memory exhaustion test should complete within timeout"
        );
    }

    #[tokio::test]
    async fn test_task_manager_circuit_breaker_integration() {
        let test_timeout = Duration::from_secs(5);
        let result = tokio::time::timeout(test_timeout, async {
            let (task_manager, mut completion_receiver) = GradientTaskManager::for_gpu_gradients(1);

            task_manager.circuit_breaker.force_open();

            let gradient_tensor = create_test_tensor(vec![2, 2]);
            let task = GradientTask::Add {
                output_grad: gradient_tensor,
                target_variable: next_variable_id(),
                operation_id: next_operation_id(),
            };

            let result = task_manager.submit_task(task).await;
            assert!(
                result.is_err(),
                "Task should be rejected by circuit breaker"
            );

            let error_msg = result.unwrap_err();
            assert!(
                error_msg.contains("Circuit breaker is open"),
                "Error should mention circuit breaker: {}",
                error_msg
            );

            match tokio::time::timeout(Duration::from_millis(500), completion_receiver.changed())
                .await
            {
                Ok(Ok(())) => {
                    let completion = completion_receiver.borrow();
                    assert!(completion.is_failed(), "Completion should indicate failure");
                    if let Some(error) = completion.get_error_message() {
                        assert!(
                            error.contains("Circuit breaker"),
                            "Completion error should mention circuit breaker"
                        );
                    }
                }
                Ok(Err(_)) => {
                    println!("Completion receiver closed");
                }
                Err(_) => {
                    println!("No completion received within timeout (this may be expected)");
                }
            }

            let cb_stats = task_manager.circuit_breaker.get_stats();
            assert!(
                cb_stats.rejected_requests > 0,
                "Should have rejected requests"
            );
        })
        .await;

        assert!(
            result.is_ok(),
            "Circuit breaker integration test should complete within timeout"
        );
    }

    #[tokio::test]
    async fn test_buffer_info_lifecycle() {
        use wgpu::{BufferDescriptor, BufferUsages};

        let (device, _) = create_test_gpu_context().await;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Test Buffer"),
            size: 1024,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let buffer_info = BufferInfo::new(
            buffer,
            1024,
            BufferUsages::STORAGE,
            AllocationStrategy::Pooled,
            true,
        );

        assert_eq!(buffer_info.get_ref_count(), 1);

        let new_count = buffer_info.increment_ref();
        assert_eq!(new_count, 2);
        assert_eq!(buffer_info.get_ref_count(), 2);

        let decremented_count = buffer_info.decrement_ref();
        assert_eq!(decremented_count, 1);
        assert_eq!(buffer_info.get_ref_count(), 1);

        assert!(!buffer_info.is_cleanup_candidate(60, 30));

        buffer_info.touch();
        let age = buffer_info.age();
        let idle_time = buffer_info.seconds_since_last_use();

        println!("Buffer age: {:?}, idle time: {}s", age, idle_time);
        assert!(idle_time < 5);
    }

    #[tokio::test]
    async fn test_gradient_task_complexity_estimation() {
        let small_tensor = create_test_tensor(vec![10, 10]);
        let large_tensor = create_test_tensor(vec![100, 100]);

        let add_task = GradientTask::Add {
            output_grad: small_tensor.clone(),
            target_variable: next_variable_id(),
            operation_id: next_operation_id(),
        };

        let matmul_task = GradientTask::MatMul {
            output_grad: small_tensor.clone(),
            other_input: large_tensor.clone(),
            transpose_other: false,
            target_variable: next_variable_id(),
            operation_id: next_operation_id(),
        };

        let add_complexity = add_task.estimate_complexity();
        let matmul_complexity = matmul_task.estimate_complexity();

        assert_eq!(add_complexity, 100);
        assert!(
            matmul_complexity > add_complexity,
            "MatMul should be more complex than Add: {} vs {}",
            matmul_complexity,
            add_complexity
        );

        assert_eq!(add_task.get_task_type(), "Add");
        assert_eq!(matmul_task.get_task_type(), "MatMul");
    }

    #[tokio::test]
    async fn test_resource_guard_integration() {
        let limits = ResourceLimits::generic();
        let tracker = ResourceTracker::new(limits).expect("Should create tracker");

        let operation_guard = tracker
            .reserve_operation()
            .expect("Should reserve operation");

        let usage_before = tracker.current_usage();
        assert_eq!(usage_before.gpu_operations, 1);
        assert_eq!(usage_before.total_queue_depth, 1);

        let (queue_guard, gpu_guard) = operation_guard.split();

        let usage_after_split = tracker.current_usage();
        assert_eq!(usage_after_split.gpu_operations, 1);
        assert_eq!(usage_after_split.total_queue_depth, 1);

        drop(queue_guard);
        let usage_after_queue_drop = tracker.current_usage();
        assert_eq!(usage_after_queue_drop.gpu_operations, 1);
        assert_eq!(usage_after_queue_drop.total_queue_depth, 0);

        drop(gpu_guard);
        let usage_final = tracker.current_usage();
        assert_eq!(usage_final.gpu_operations, 0);
        assert_eq!(usage_final.total_queue_depth, 0);
    }

    #[test]
    fn test_task_completion_helpers() {
        let operation_id = next_operation_id();
        let variable_id = next_variable_id();
        let duration = Duration::from_millis(100);

        let success = TaskCompletion::success(operation_id, variable_id, duration, Some(0.5));
        assert!(success.is_success());
        assert!(!success.is_failed());
        assert!(!success.is_system_failure());
        assert!(success.get_error_message().is_none());
        assert!(!success.has_gradient_issues());

        let failed = TaskCompletion::failed(
            operation_id,
            variable_id,
            duration,
            "Test error".to_string(),
        );
        assert!(!failed.is_success());
        assert!(failed.is_failed());
        assert!(!failed.is_system_failure());
        assert_eq!(failed.get_error_message(), Some("Test error"));

        let timeout = TaskCompletion::timed_out(operation_id, variable_id, duration);
        assert!(!timeout.is_success());
        assert!(timeout.is_failed());
        assert!(timeout.is_system_failure());
        assert_eq!(timeout.get_error_message(), Some("Operation timed out"));

        let gradient_issues =
            TaskCompletion::success(operation_id, variable_id, duration, Some(f32::NAN));
        assert!(gradient_issues.has_gradient_issues());

        let vanishing_gradient =
            TaskCompletion::success(operation_id, variable_id, duration, Some(1e-8));
        assert!(vanishing_gradient.has_gradient_issues());
    }
}
