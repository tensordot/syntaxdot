use std::fmt::Debug;

use tch::Tensor;

/// Module for which a computation can fail.
pub trait FallibleModule: Debug + Send {
    /// The error type.
    type Error;

    /// Apply the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error>;
}

/// Module for which a computation can fail.
pub trait FallibleModuleT: Debug + Send {
    /// The error type.
    type Error;

    /// Apply the module.
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error>;
}

impl<M> FallibleModuleT for M
where
    M: FallibleModule,
{
    type Error = M::Error;

    fn forward_t(&self, input: &Tensor, _train: bool) -> Result<Tensor, Self::Error> {
        self.forward(input)
    }
}
