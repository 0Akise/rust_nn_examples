use super::autograd::{GpuContext, Variable};

use gpu_accel::{Shape, Tensor};

use std::collections::{HashMap, HashSet, VecDeque};

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
    id_next: usize,
}

impl ExprGraph {
    pub fn new() -> Self {
        return Self {
            nodes: HashMap::new(),
            inputs: HashMap::new(),
            id_next: 0,
        };
    }

    fn next_id(&mut self) -> ExprId {
        let id = self.id_next;

        self.id_next += 1;

        return id;
    }

    pub fn input(&mut self, var: Variable) -> ExprId {
        let id = self.next_id();

        self.inputs.insert(id, var);

        return id;
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

        return Ok(id);
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

        return Ok(id);
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

        return Ok(id);
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

        return Ok(id);
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

        return Ok(id);
    }

    fn get_shape(&self, id: ExprId) -> Result<Shape, String> {
        if let Some(var) = self.inputs.get(&id) {
            return Ok(var.tensor.shape.clone());
        } else if let Some(node) = self.nodes.get(&id) {
            return node
                .shape
                .clone()
                .ok_or("Node missing shape information".to_string());
        } else {
            return Err("Node not found in graph".to_string());
        }
    }

    pub fn num_nodes(&self) -> usize {
        return self.nodes.len();
    }

    pub fn debug_print(&self) {
        println!("🔍 Graph Debug Info:");
        println!(
            "  Input nodes: {:?}",
            self.inputs.keys().collect::<Vec<_>>()
        );
        println!("  Computation nodes:");
        for (id, node) in &self.nodes {
            println!(
                "    Node {}: op={:?}, inputs={:?}",
                id, node.op, node.inputs
            );
        }
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

    fn canonical_topological_sort(
        &self,
        graph: &ExprGraph,
        id_output: ExprId,
    ) -> Result<Vec<ExprId>, Box<dyn std::error::Error>> {
        let mut reachable = HashSet::new();
        let mut to_visit = VecDeque::new();

        to_visit.push_back(id_output);

        while let Some(node_id) = to_visit.pop_front() {
            if reachable.contains(&node_id) {
                continue;
            }
            reachable.insert(node_id);

            if let Some(node) = graph.nodes.get(&node_id) {
                for &input_id in &node.inputs {
                    if !reachable.contains(&input_id) {
                        to_visit.push_back(input_id);
                    }
                }
            }
        }

        let mut in_degree: HashMap<ExprId, usize> = HashMap::new();

        for &node_id in &reachable {
            in_degree.insert(node_id, 0);
        }

        for &node_id in &reachable {
            if let Some(node) = graph.nodes.get(&node_id) {
                for &input_id in &node.inputs {
                    if reachable.contains(&input_id) {
                        *in_degree.get_mut(&node_id).unwrap() += 1;
                    }
                }
            }
        }

        let mut queue = std::collections::BinaryHeap::new();
        let mut result = Vec::new();

        for &node_id in &reachable {
            if in_degree[&node_id] == 0 {
                let priority = if graph.inputs.contains_key(&node_id) {
                    -(node_id as i32) - 1000
                } else {
                    -(node_id as i32)
                };

                queue.push((priority, node_id));
            }
        }

        while let Some((_, node_id)) = queue.pop() {
            result.push(node_id);

            for &dependent_id in &reachable {
                if let Some(dependent_node) = graph.nodes.get(&dependent_id) {
                    if dependent_node.inputs.contains(&node_id) {
                        let count = in_degree.get_mut(&dependent_id).unwrap();

                        *count -= 1;

                        if *count == 0 {
                            let priority = -(dependent_id as i32);

                            queue.push((priority, dependent_id));
                        }
                    }
                }
            }
        }

        if result.len() != reachable.len() {
            println!(
                "⚠️ Result length {} != reachable length {}",
                result.len(),
                reachable.len()
            );
            println!(
                "⚠️ Missing nodes: {:?}",
                reachable
                    .difference(&result.iter().cloned().collect())
                    .collect::<Vec<_>>()
            );
            return Err("Cycle detected in expression graph".into());
        }

        return Ok(result);
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

        let execution_order = self.canonical_topological_sort(graph, output_id)?;

        for node_id in execution_order {
            if computed.contains_key(&node_id) {
                continue;
            }

            let node = graph
                .nodes
                .get(&node_id)
                .ok_or_else(|| format!("Node {} not found in computation graph", node_id))?;

            let mut input_vars = Vec::new();

            for &input_id in &node.inputs {
                let input_var = computed.get(&input_id).ok_or_else(|| {
                    format!(
                        "Input {} not computed when needed for node {} (op: {:?}). Available: {:?}",
                        input_id,
                        node_id,
                        node.op,
                        computed.keys().collect::<Vec<_>>()
                    )
                })?;

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

        return Ok(computed
            .get(&output_id)
            .ok_or("Output node not computed")?
            .clone());
    }

    pub async fn backward(
        &mut self,
        mut result: Variable,
    ) -> Result<Variable, Box<dyn std::error::Error>> {
        self.context.backward(&mut result);

        return Ok(result);
    }
}
