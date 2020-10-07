use tch::nn::VarStore;
use tch::Tensor;

mod adamw;
pub use adamw::{AdamW, AdamWConfig};

mod grad;
pub use grad::ZeroGrad;

mod grad_scale;
pub use grad_scale::GradScaler;

pub trait Optimizer {
    type Config;

    /// Perform a backward step on the given loss.
    ///
    /// The provided configuration function is given the full name of
    /// the variable and should return the Adam configuration. This
    /// makes it possible to use different Adam hyper parameters for
    /// different parts of a model.
    fn backward_step<F>(&mut self, loss: &Tensor, config_fun: F)
    where
        F: Fn(&str) -> Self::Config;

    /// Perform an update step.
    ///
    /// The provided configuration function is given the full name of
    /// the variable and should return the Adam configuration. This
    /// makes it possible to use different Adam hyper parameters for
    /// different parts of a model.
    ///
    /// It is generally recommended to use `backward_step`, since it
    /// computes the gradients and performs any loss scaling (if
    /// necessary).
    fn step<F>(&mut self, config_fun: F)
    where
        F: Fn(&str) -> Self::Config;

    /// Get the variable store used by the optimizer.
    fn var_store(&self) -> &VarStore;
}
