use tch::nn::{self};
use tch::Tensor;

mod grad;
pub use grad::ZeroGrad;

mod grad_scale;
use crate::error::SyntaxDotError;
pub use grad_scale::GradScaler;

pub trait Optimizer {
    /// Perform a backward step on the given loss.
    fn backward_step(&mut self, loss: &Tensor) -> Result<(), SyntaxDotError>;

    /// Set the learning rate for a parameter group.
    fn set_lr_group(&mut self, group: usize, learning_rate: f64);

    /// Set the weight decay for a parameter group.
    fn set_weight_decay_group(&mut self, group: usize, weight_decay: f64);

    /// Perform an update step.
    ///
    /// It is generally recommended to use `backward_step`, since it
    /// computes the gradients and performs any loss scaling (if
    /// necessary).
    fn step(&mut self);

    /// Get the trainable variables associated with the optimizer.
    fn trainable_variables(&self) -> Vec<Tensor>;
}

impl Optimizer for nn::Optimizer {
    fn backward_step(&mut self, loss: &Tensor) -> Result<(), SyntaxDotError> {
        nn::Optimizer::backward_step(self, loss);
        Ok(())
    }

    fn set_lr_group(&mut self, group: usize, learning_rate: f64) {
        nn::Optimizer::set_lr_group(self, group, learning_rate)
    }

    fn set_weight_decay_group(&mut self, group: usize, weight_decay: f64) {
        nn::Optimizer::set_weight_decay_group(self, group, weight_decay)
    }

    fn step(&mut self) {
        nn::Optimizer::step(self)
    }

    fn trainable_variables(&self) -> Vec<Tensor> {
        nn::Optimizer::trainable_variables(self)
    }
}
