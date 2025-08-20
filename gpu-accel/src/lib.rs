pub mod gpu_module;
pub mod shader_manager;

use gpu_module::GpuModule;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

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

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape: Shape,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.total_elements());
        Self {
            data: Arc::new(data),
            shape,
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = vec![0.0; shape.total_elements()];
        Self::new(data, shape)
    }

    pub fn ones(shape: Shape) -> Self {
        let data = vec![1.0; shape.total_elements()];
        Self::new(data, shape)
    }

    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        Self::new(data, shape)
    }

    pub fn data_ref(&self) -> &[f32] {
        &self.data
    }

    pub fn data_owned(&self) -> Vec<f32> {
        (*self.data).clone()
    }

    pub fn add_inplace(&mut self, other: &Tensor) -> Result<(), String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape.dims, other.shape.dims
            ));
        }

        let mut new_data = self.data_owned();
        for (a, b) in new_data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }

        self.data = Arc::new(new_data);
        Ok(())
    }

    pub fn add_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape.dims, other.shape.dims
            ));
        }

        let result_data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor::new(result_data, self.shape.clone()))
    }

    pub fn to_gpu_format(&self) -> Vec<TensorElement> {
        self.data
            .iter()
            .map(|&x| TensorElement { value: x })
            .collect()
    }

    pub fn shares_data_with(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    pub fn reference_count(&self) -> usize {
        Arc::strong_count(&self.data)
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
    Mul,
    Add,
    MatMul,
    Dot,
    Transpose,
    Reduce(ReduceOp),
}

pub struct GpuSession {
    pub gpu: GpuModule,
}

impl GpuSession {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        return Ok(Self {
            gpu: GpuModule::new().await?,
        });
    }

    pub async fn add(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::Add).await;
    }

    pub async fn mul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::Mul).await;
    }

    pub async fn matmul(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::MatMul).await;
    }

    pub async fn dot(
        &mut self,
        a: &Tensor,
        b: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.binary_op(a, b, Operation::Dot).await;
    }

    pub async fn transpose(
        &mut self,
        tensor: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        return self.gpu.unary_op(tensor, Operation::Transpose).await;
    }

    pub async fn batch_operations<F, R>(&mut self, operations: F) -> R
    where
        F: FnOnce(&mut GpuModule) -> R,
    {
        return operations(&mut self.gpu);
    }
}
