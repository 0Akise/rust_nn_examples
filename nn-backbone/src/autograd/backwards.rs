use super::{BackwardFn, Variable};

use std::sync::{Arc, Mutex, Weak};

use gpu_accel::{GpuSession, Tensor};

fn accumulate_gradient(variable: &mut Variable, new_grad: &Tensor) {
    match &mut variable.grad {
        Some(existing_grad) => {
            *existing_grad = Tensor::new(new_grad.data.clone(), new_grad.shape.clone());
        }

        None => {
            variable.grad = Some(Tensor::new(new_grad.data.clone(), new_grad.shape.clone()));
        }
    }
}

async fn broadcast_multiply(
    scalar: &Tensor,
    vector: &Tensor,
    session: &mut GpuSession,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let scalar_value = scalar.data[0];
    let expanded_scalar = Tensor::new(
        vec![scalar_value; vector.shape.total_elements()],
        vector.shape.clone(),
    );

    return session.mul(&expanded_scalar, vector).await;
}

pub struct BackwardAdd {
    pub input_a: Weak<Mutex<Variable>>,
    pub input_b: Weak<Mutex<Variable>>,
    pub session: Arc<Mutex<GpuSession>>,
}

impl BackwardFn for BackwardAdd {
    fn backward(&self, grad_output: &Tensor) {
        if let Some(input_a) = self.input_a.upgrade() {
            if let Ok(mut var_a) = input_a.try_lock() {
                if var_a.requires_grad {
                    accumulate_gradient(&mut var_a, grad_output);

                    if let Some(grad_fn) = &var_a.grad_fn {
                        if let Some(grad) = &var_a.grad {
                            grad_fn.backward(grad);
                        }
                    }
                }
            }
        }

        if let Some(input_b) = self.input_b.upgrade() {
            if let Ok(mut var_b) = input_b.try_lock() {
                if var_b.requires_grad {
                    accumulate_gradient(&mut var_b, grad_output);

                    if let Some(grad_fn) = &var_b.grad_fn {
                        if let Some(grad) = &var_b.grad {
                            grad_fn.backward(grad);
                        }
                    }
                }
            }
        }
    }
}

pub struct BackwardMul {
    pub input_a: Weak<Mutex<Variable>>,
    pub input_b: Weak<Mutex<Variable>>,
    pub input_a_data: Tensor,
    pub input_b_data: Tensor,
    pub session: Arc<Mutex<GpuSession>>,
}

impl BackwardFn for BackwardMul {
    fn backward(&self, grad_output: &Tensor) {
        if let Some(input_a) = self.input_a.upgrade() {
            if let Ok(mut var_a) = input_a.try_lock() {
                if var_a.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(grad_a) =
                            pollster::block_on(session.mul(grad_output, &self.input_b_data))
                        {
                            accumulate_gradient(&mut var_a, &grad_a);

                            if let Some(grad_fn) = &var_a.grad_fn {
                                if let Some(grad) = &var_a.grad {
                                    grad_fn.backward(grad);
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(input_b) = self.input_b.upgrade() {
            if let Ok(mut var_b) = input_b.try_lock() {
                if var_b.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(grad_b) =
                            pollster::block_on(session.mul(grad_output, &self.input_a_data))
                        {
                            accumulate_gradient(&mut var_b, &grad_b);

                            if let Some(grad_fn) = &var_b.grad_fn {
                                if let Some(grad) = &var_b.grad {
                                    grad_fn.backward(grad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct BackwardMatMul {
    pub input_a: Weak<Mutex<Variable>>,
    pub input_b: Weak<Mutex<Variable>>,
    pub input_a_data: Tensor,
    pub input_b_data: Tensor,
    pub session: Arc<Mutex<GpuSession>>,
}

impl BackwardFn for BackwardMatMul {
    fn backward(&self, grad_output: &Tensor) {
        if let Some(input_a) = self.input_a.upgrade() {
            if let Ok(mut var_a) = input_a.try_lock() {
                if var_a.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(b_transposed) =
                            pollster::block_on(session.transpose(&self.input_b_data))
                        {
                            if let Ok(grad_a) =
                                pollster::block_on(session.matmul(grad_output, &b_transposed))
                            {
                                accumulate_gradient(&mut var_a, &grad_a);

                                if let Some(grad_fn) = &var_a.grad_fn {
                                    if let Some(grad) = &var_a.grad {
                                        grad_fn.backward(grad);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(input_b) = self.input_b.upgrade() {
            if let Ok(mut var_b) = input_b.try_lock() {
                if var_b.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(a_transposed) =
                            pollster::block_on(session.transpose(&self.input_a_data))
                        {
                            if let Ok(grad_b) =
                                pollster::block_on(session.matmul(&a_transposed, grad_output))
                            {
                                accumulate_gradient(&mut var_b, &grad_b);

                                if let Some(grad_fn) = &var_b.grad_fn {
                                    if let Some(grad) = &var_b.grad {
                                        grad_fn.backward(grad);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct BackwardDot {
    pub input_a: Weak<Mutex<Variable>>,
    pub input_b: Weak<Mutex<Variable>>,
    pub input_a_data: Tensor,
    pub input_b_data: Tensor,
    pub session: Arc<Mutex<GpuSession>>,
}

impl BackwardFn for BackwardDot {
    fn backward(&self, grad_output: &Tensor) {
        if let Some(input_a) = self.input_a.upgrade() {
            if let Ok(mut var_a) = input_a.try_lock() {
                if var_a.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(grad_a) = pollster::block_on(broadcast_multiply(
                            grad_output,
                            &self.input_b_data,
                            &mut *session,
                        )) {
                            accumulate_gradient(&mut var_a, &grad_a);

                            if let Some(grad_fn) = &var_a.grad_fn {
                                if let Some(grad) = &var_a.grad {
                                    grad_fn.backward(grad);
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(input_b) = self.input_b.upgrade() {
            if let Ok(mut var_b) = input_b.try_lock() {
                if var_b.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(grad_b) = pollster::block_on(broadcast_multiply(
                            grad_output,
                            &self.input_a_data,
                            &mut *session,
                        )) {
                            accumulate_gradient(&mut var_b, &grad_b);

                            if let Some(grad_fn) = &var_b.grad_fn {
                                if let Some(grad) = &var_b.grad {
                                    grad_fn.backward(grad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub struct BackwardTranspose {
    pub input: Weak<Mutex<Variable>>,
    pub session: Arc<Mutex<GpuSession>>,
}

impl BackwardFn for BackwardTranspose {
    fn backward(&self, grad_output: &Tensor) {
        if let Some(input) = self.input.upgrade() {
            if let Ok(mut var) = input.try_lock() {
                if var.requires_grad {
                    if let Ok(mut session) = self.session.try_lock() {
                        if let Ok(grad_input) = pollster::block_on(session.transpose(grad_output)) {
                            accumulate_gradient(&mut var, &grad_input);

                            if let Some(grad_fn) = &var.grad_fn {
                                if let Some(grad) = &var.grad {
                                    grad_fn.backward(grad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
