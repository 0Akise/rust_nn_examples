use std::collections::HashMap;

use super::{Operation, Shape};

pub struct ShaderTemplate {
    pub template: String,
    pub variables: HashMap<String, String>,
}

impl ShaderTemplate {
    pub fn set_variable<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: Into<String>,
        V: std::fmt::Display,
    {
        self.variables.insert(key.into(), value.to_string());
        self
    }

    pub fn render(&self) -> String {
        let mut result = self.template.clone();

        for (key, value) in &self.variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }
}

pub struct ShaderManager {
    template_cache: HashMap<Operation, String>,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self {
            template_cache: HashMap::new(),
        }
    }

    pub fn load_templates(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.template_cache.insert(
            Operation::ElementWiseMultiply,
            include_str!("shaders/elementwise_multiply.wgsl").to_string(),
        );

        self.template_cache.insert(
            Operation::ElementWiseAdd,
            include_str!("shaders/elementwise_add.wgsl").to_string(),
        );

        self.template_cache.insert(
            Operation::MatrixMultiply,
            include_str!("shaders/matrix_multiply.wgsl").to_string(),
        );

        self.template_cache.insert(
            Operation::Transpose,
            include_str!("shaders/transpose.wgsl").to_string(),
        );

        Ok(())
    }

    pub fn generate_shader_source(
        &self,
        op: &Operation,
        shape_a: &Shape,
        shape_b: Option<&Shape>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let template_source = self
            .template_cache
            .get(op)
            .ok_or("Template not found for operation")?;

        let mut template = ShaderTemplate {
            template: template_source.clone(),
            variables: HashMap::new(),
        };

        self.set_template_variables(&mut template, op, shape_a, shape_b);

        Ok(template.render())
    }

    fn set_template_variables(
        &self,
        template: &mut ShaderTemplate,
        op: &Operation,
        shape_a: &Shape,
        shape_b: Option<&Shape>,
    ) {
        match op {
            Operation::ElementWiseMultiply | Operation::ElementWiseAdd => {
                template.set_variable("TOTAL_ELEMENTS", shape_a.total_elements());
                template.set_variable("WORKGROUP_SIZE", 64);
            }

            Operation::MatrixMultiply => {
                let shape_b = shape_b.expect("Matrix multiply requires two shapes");
                template
                    .set_variable("M", shape_a.dims[0])
                    .set_variable("N", shape_a.dims[1])
                    .set_variable("P", shape_b.dims[1])
                    .set_variable("WORKGROUP_SIZE_X", 8)
                    .set_variable("WORKGROUP_SIZE_Y", 8);
            }

            Operation::Transpose => {
                assert_eq!(shape_a.rank(), 2, "Transpose only supports 2D matrices");
                template
                    .set_variable("ROWS", shape_a.dims[0])
                    .set_variable("COLS", shape_a.dims[1])
                    .set_variable("WORKGROUP_SIZE", 8); // 8x8 workgroup for 2D operations
            }

            _ => {}
        }
    }
}
