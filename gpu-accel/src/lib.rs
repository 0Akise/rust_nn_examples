pub mod gpu_module;
pub mod shader_manager;

use bytemuck::{Pod, Zeroable};

pub use gpu_module::GpuModule;
pub use shader_manager::{ShaderManager, ShaderTemplate};

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn flatten_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.dims.len());
        let mut flat_index = 0;
        let mut stride = 1;

        for i in (0..self.dims.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= self.dims[i];
        }
        flat_index
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TensorElement {
    pub value: f32,
}

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.total_elements());
        Self { data, shape }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.total_elements()];
        Self { data, shape }
    }

    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.0; shape.total_elements()];
        Self { data, shape }
    }

    pub fn to_gpu_format(&self) -> Vec<TensorElement> {
        self.data
            .iter()
            .map(|&x| TensorElement { value: x })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    ElementWiseMultiply,
    ElementWiseAdd,
    MatrixMultiply,
    Transpose,
    Reduce(ReduceOp),
}

pub struct GpuSession {
    gpu: GpuModule,
}

impl GpuSession {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        return Ok(Self {
            gpu: GpuModule::new().await?,
        });
    }

    pub async fn matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::MatrixMultiply).await;
    }

    pub async fn add(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::ElementWiseAdd).await;
    }

    pub async fn multiply(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self
            .gpu
            .binary_op(a, b, Operation::ElementWiseMultiply)
            .await;
    }

    pub async fn batch_operations<F, R>(&mut self, operations: F) -> R
    where
        F: FnOnce(&mut GpuModule) -> R,
    {
        return operations(&mut self.gpu);
    }
}
