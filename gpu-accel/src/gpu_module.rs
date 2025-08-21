use crate::shader_manager::ShaderManager;
use crate::{Operation, Shape, Tensor, TensorElement};

use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use wgpu::{Adapter, BindGroupLayout, ComputePipeline, Device, Queue};

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
}

pub struct GpuModule {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub info: GpuInfo,
    pub shader_manager: ShaderManager,
    pipeline_cache: HashMap<(Operation, Shape, Option<Shape>), Arc<ComputePipeline>>,
    bind_group_layout_cache: HashMap<Operation, Arc<BindGroupLayout>>,
}

impl GpuModule {
    pub async fn new() -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()
            .unwrap();

        let adapter_info = adapter.get_info();

        println!("Using adapter: {}", adapter_info.name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to request device: {}", e))?;

        let info = GpuInfo {
            name: adapter_info.name.clone(),
            vendor: format!("{:?}", adapter_info.vendor),
            device_type: format!("{:?}", adapter_info.device_type),
            backend: format!("{:?}", adapter_info.backend),
        };

        let mut shader_manager = ShaderManager::new();
        shader_manager.load_templates()?;

        return Ok(Self {
            adapter,
            device,
            queue,
            info,
            shader_manager,
            pipeline_cache: HashMap::new(),
            bind_group_layout_cache: HashMap::new(),
        });
    }

    pub async fn unary_op(
        &mut self,
        tensor: &Tensor,
        op: Operation,
    ) -> Result<Tensor, Box<dyn Error>> {
        let output_shape = match op {
            Operation::Transpose => match tensor.shape.rank() {
                1 => tensor.shape.clone(),
                2 => Shape::new(vec![tensor.shape.dims[1], tensor.shape.dims[0]]),
                _ => {
                    return Err(format!(
                        "Transpose only supports 1D/2Ds. got {}D tensor",
                        tensor.shape.rank()
                    )
                    .into());
                }
            },
            _ => return Err("Unary operation not supported".into()),
        };

        if let Operation::Transpose = op {
            if tensor.shape.rank() == 1 {
                return Ok(Tensor::new(tensor.data.to_vec(), output_shape));
            }
        }

        let shader_source = self
            .shader_manager
            .generate_shader_source(&op, &tensor.shape, None)?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{:?} Shader", op)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Unary Op Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let input_data = tensor.to_gpu_format();

        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unary Op Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Unary Op Compute Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Unary Op Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unary Op Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Unary Op Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            match op {
                Operation::Transpose => {
                    let workgroups_x = (tensor.shape.dims[0] + 7) / 8;
                    let workgroups_y = (tensor.shape.dims[1] + 7) / 8;
                    compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
                }
                _ => {
                    return Err("Unsupported unary operation".into());
                }
            }
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        loop {
            self.device
                .poll(wgpu::PollType::Wait)
                .expect("Failed to poll device");

            if let Ok(result) = receiver.try_recv() {
                result?;
                break;
            }
        }

        let data = buffer_slice.get_mapped_range();
        let result: &[TensorElement] = bytemuck::cast_slice(&data);
        let output_data: Vec<f32> = result.iter().map(|x| x.value).collect();

        drop(data);

        staging_buffer.unmap();

        return Ok(Tensor::new(output_data, output_shape));
    }

    pub async fn binary_op(
        &mut self,
        tensor_a: &Tensor,
        tensor_b: &Tensor,
        op: Operation,
    ) -> Result<Tensor, Box<dyn Error>> {
        if tensor_a.shape.total_elements() == 0 || tensor_b.shape.total_elements() == 0 {
            return Err(format!(
                "Cannot operate on empty tensors: a={}, b={}",
                tensor_a.shape.total_elements(),
                tensor_b.shape.total_elements()
            )
            .into());
        }

        const MAX_ELEMENTS: usize = 10_000_000;

        let total_elements = tensor_a.shape.total_elements() + tensor_b.shape.total_elements();

        if total_elements > MAX_ELEMENTS {
            return Err(format!(
                "Tensor operation too large: {} elements (max {})",
                total_elements, MAX_ELEMENTS
            )
            .into());
        }

        let output_shape = match op {
            Operation::Add | Operation::Mul => {
                assert_eq!(tensor_a.shape, tensor_b.shape);
                tensor_a.shape.clone()
            }
            Operation::MatMul => {
                assert_eq!(tensor_a.shape.rank(), 2);
                assert_eq!(tensor_b.shape.rank(), 2);
                assert_eq!(tensor_a.shape.dims[1], tensor_b.shape.dims[0]);
                Shape::new(vec![tensor_a.shape.dims[0], tensor_b.shape.dims[1]])
            }
            Operation::Dot => {
                assert_eq!(tensor_a.shape.rank(), 1);
                assert_eq!(tensor_b.shape.rank(), 1);
                assert_eq!(tensor_a.shape, tensor_b.shape);
                Shape::new(vec![1])
            }
            _ => return Err("Operation not supported".into()),
        };

        let shader_source = self.shader_manager.generate_shader_source(
            &op,
            &tensor_a.shape,
            Some(&tensor_b.shape),
        )?;

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{:?} Shader", op)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Binary Op Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let input_a_data = tensor_a.to_gpu_format();
        let input_b_data = tensor_b.to_gpu_format();

        let input_a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input A Buffer"),
                contents: bytemuck::cast_slice(&input_a_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let input_b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input B Buffer"),
                contents: bytemuck::cast_slice(&input_b_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Op Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Binary Op Compute Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Binary Op Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Binary Op Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Op Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            match op {
                Operation::MatMul => {
                    let workgroups_x = (output_shape.dims[0] + 7) / 8;
                    let workgroups_y = (output_shape.dims[1] + 7) / 8;
                    compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
                }
                _ => {
                    let workgroups = (output_shape.total_elements() + 63) / 64;
                    compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
                }
            }
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_shape.total_elements() * std::mem::size_of::<TensorElement>()) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        loop {
            self.device
                .poll(wgpu::PollType::Wait)
                .expect("Failed to poll device");

            if let Ok(result) = receiver.try_recv() {
                result?;
                break;
            }
        }

        let data = buffer_slice.get_mapped_range();
        let result: &[TensorElement] = bytemuck::cast_slice(&data);
        let output_data: Vec<f32> = result.iter().map(|x| x.value).collect();

        drop(data);
        staging_buffer.unmap();

        return Ok(Tensor::new(output_data, output_shape));
    }

    pub async fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        return self.binary_op(a, b, Operation::Add).await;
    }

    pub async fn mul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        return self.binary_op(a, b, Operation::Mul).await;
    }

    pub async fn matmul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        return self.binary_op(a, b, Operation::MatMul).await;
    }

    pub async fn dot(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        return self.binary_op(a, b, Operation::Dot).await;
    }

    pub async fn transpose(&mut self, t: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        return self.unary_op(t, Operation::Transpose).await;
    }

    pub fn print_info(&self) {
        println!("GPU Info:");
        println!("  Name: {}", self.info.name);
        println!("  Vendor: {}", self.info.vendor);
        println!("  Type: {}", self.info.device_type);
        println!("  Backend: {}", self.info.backend);
    }
}
