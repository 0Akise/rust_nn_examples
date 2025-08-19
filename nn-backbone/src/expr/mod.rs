use super::autograd::{GpuContext, Variable};

use gpu_accel::{Shape, Tensor};

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Mul,
    MatMul,
    Dot,
    Transpose,
    ReLU,
}

pub type ExprId = usize;

#[derive(Debug, Clone)]
pub struct ExprNode {
    pub id: usize,
    pub op: Op,
    pub inputs: Vec<ExprId>,
    pub shape: Option<Shape>,
}

pub struct ExprGraph {
    nodes: HashMap<ExprId, ExprNode>,
    inputs: HashMap<ExprId, Variable>,
    next_id: usize,
}

impl ExprGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            inputs: HashMap::new(),
            next_id: 0,
        }
    }

    fn next_id(&mut self) -> ExprId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn input(&mut self, var: Variable) -> ExprId {
        let id = self.next_id();
        self.inputs.insert(id, var);
        id
    }

    pub fn add(&mut self, a: ExprId, b: ExprId) -> Result<ExprId, String> {
        let a_shape = self.get_shape(a)?;
        let b_shape = self.get_shape(b)?;

        if a_shape != b_shape {
            return Err(format!(
                "Shape mismatch in add: {:?} vs {:?}",
                a_shape.dims, b_shape.dims
            ));
        }

        let id = self.next_id();
        self.nodes.insert(
            id,
            ExprNode {
                id,
                op: Op::Add,
                inputs: vec![a, b],
                shape: Some(a_shape),
            },
        );

        Ok(id)
    }

    pub fn mul(&mut self, a: ExprId, b: ExprId) -> Result<ExprId, String> {
        let a_shape = self.get_shape(a)?;
        let b_shape = self.get_shape(b)?;

        if a_shape != b_shape {
            return Err(format!(
                "Shape mismatch in mul: {:?} vs {:?}",
                a_shape.dims, b_shape.dims
            ));
        }

        let id = self.next_id();
        self.nodes.insert(
            id,
            ExprNode {
                id,
                op: Op::Mul,
                inputs: vec![a, b],
                shape: Some(a_shape),
            },
        );

        Ok(id)
    }

    pub fn matmul(&mut self, a: ExprId, b: ExprId) -> Result<ExprId, String> {
        let a_shape = self.get_shape(a)?;
        let b_shape = self.get_shape(b)?;

        if a_shape.rank() != 2 || b_shape.rank() != 2 {
            return Err("MatMul requires 2D tensors".to_string());
        }

        if a_shape.dims[1] != b_shape.dims[0] {
            return Err(format!(
                "MatMul dimension mismatch: {} vs {}",
                a_shape.dims[1], b_shape.dims[0]
            ));
        }

        let output_shape = Shape::new(vec![a_shape.dims[0], b_shape.dims[1]]);
        let id = self.next_id();

        self.nodes.insert(
            id,
            ExprNode {
                id,
                op: Op::MatMul,
                inputs: vec![a, b],
                shape: Some(output_shape),
            },
        );

        Ok(id)
    }

    pub fn relu(&mut self, x: ExprId) -> Result<ExprId, String> {
        let x_shape = self.get_shape(x)?;
        let id = self.next_id();

        self.nodes.insert(
            id,
            ExprNode {
                id,
                op: Op::ReLU,
                inputs: vec![x],
                shape: Some(x_shape),
            },
        );

        Ok(id)
    }

    pub fn transpose(&mut self, x: ExprId) -> Result<ExprId, String> {
        let x_shape = self.get_shape(x)?;

        if x_shape.rank() != 2 {
            return Err("Transpose only supports 2D tensors".to_string());
        }

        let output_shape = Shape::new(vec![x_shape.dims[1], x_shape.dims[0]]);
        let id = self.next_id();

        self.nodes.insert(
            id,
            ExprNode {
                id,
                op: Op::Transpose,
                inputs: vec![x],
                shape: Some(output_shape),
            },
        );

        Ok(id)
    }

    fn get_shape(&self, id: ExprId) -> Result<Shape, String> {
        if let Some(var) = self.inputs.get(&id) {
            Ok(var.tensor.shape.clone())
        } else if let Some(node) = self.nodes.get(&id) {
            node.shape
                .clone()
                .ok_or("Node missing shape information".to_string())
        } else {
            Err("Node not found in graph".to_string())
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

pub struct ExprExecutor {
    context: GpuContext,
}

impl ExprExecutor {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            context: GpuContext::new().await?,
        })
    }

    fn dfs_topological(
        &self,
        graph: &ExprGraph,
        node_id: ExprId,
        visited: &mut std::collections::HashSet<ExprId>,
        temp_visited: &mut std::collections::HashSet<ExprId>,
        result: &mut Vec<ExprId>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if temp_visited.contains(&node_id) {
            return Err("Cycle detected in expression graph".into());
        }

        if visited.contains(&node_id) {
            return Ok(());
        }

        temp_visited.insert(node_id);

        if let Some(node) = graph.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.dfs_topological(graph, input_id, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(&node_id);
        visited.insert(node_id);
        result.push(node_id);

        Ok(())
    }

    fn topological_sort(
        &self,
        graph: &ExprGraph,
        id_output: ExprId,
    ) -> Result<Vec<ExprId>, Box<dyn std::error::Error>> {
        let mut visited = std::collections::HashSet::new();
        let mut temp_visited = std::collections::HashSet::new();
        let mut result = Vec::new();

        self.dfs_topological(
            graph,
            id_output,
            &mut visited,
            &mut temp_visited,
            &mut result,
        )?;

        result.reverse();
        Ok(result)
    }

    pub async fn compute(
        &mut self,
        graph: &ExprGraph,
        output_id: ExprId,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let mut computed: HashMap<ExprId, Variable> = HashMap::new();

        for (&id, var) in &graph.inputs {
            computed.insert(id, var.clone());
        }

        let execution_order = self.topological_sort(graph, output_id)?;

        for node_id in execution_order {
            if computed.contains_key(&node_id) {
                continue;
            }

            let node = graph.nodes.get(&node_id).ok_or("Node not found in graph")?;
            let mut input_vars = Vec::new();

            for &input_id in &node.inputs {
                let input_var = computed
                    .get(&input_id)
                    .ok_or("Input not computed when needed")?;

                input_vars.push(input_var.clone());
            }

            let result = match node.op {
                Op::Add => {
                    self.context
                        .forward_add(&input_vars[0], &input_vars[1])
                        .await?
                }
                Op::Mul => {
                    self.context
                        .forward_mul(&input_vars[0], &input_vars[1])
                        .await?
                }
                Op::MatMul => {
                    self.context
                        .forward_matmul(&input_vars[0], &input_vars[1])
                        .await?
                }
                Op::Dot => {
                    self.context
                        .forward_dot(&input_vars[0], &input_vars[1])
                        .await?
                }
                Op::Transpose => self.context.forward_transpose(&input_vars[0]).await?,
                Op::ReLU => {
                    let input_data: Vec<f32> = input_vars[0]
                        .tensor
                        .data
                        .iter()
                        .map(|&x| if x > 0.0 { x } else { 0.0 })
                        .collect();
                    Variable::with_grad(Tensor::new(input_data, input_vars[0].tensor.shape.clone()))
                }
            };

            computed.insert(node_id, result);
        }

        Ok(computed
            .get(&output_id)
            .ok_or("Output node not computed")?
            .clone())
    }

    pub async fn compute_with_grad(
        &mut self,
        graph: &ExprGraph,
        id_output: ExprId,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        let mut result = self.compute(graph, id_output).await?;

        self.context.backward(&mut result);

        Ok(result)
    }
}
