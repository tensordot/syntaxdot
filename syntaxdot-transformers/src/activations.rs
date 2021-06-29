//! Activation functions

use std::f64;

use tch::Tensor;

use crate::module::FallibleModule;
use crate::TransformerError;

pub trait Activation: Clone + FallibleModule {}

/// GELU activation function (Google/OpenAI flavor).
///
/// GELU(x)=x Φ(x)
///
/// where Φ(x) is the CDF for the Gaussian distribution.
#[derive(Clone, Copy, Debug)]
pub struct GeluNew;

impl FallibleModule for GeluNew {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(0.5
            * input
            * (1.0
                + Tensor::f_tanh(
                    &((2. / f64::consts::PI).sqrt() * (input + 0.044715 * input.pow(3.0))),
                )?))
    }
}

impl Activation for GeluNew {}

/// GELU activation function.
///
/// GELU(x)=x Φ(x)
///
/// where Φ(x) is the CDF for the Gaussian distribution.
#[derive(Clone, Copy, Debug)]
pub struct Gelu;

impl FallibleModule for Gelu {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(input.f_gelu()?)
    }
}

impl Activation for Gelu {}

/// ReLU activation function
///
/// ReLU(x)=max(0,x)
#[derive(Clone, Copy, Debug)]
pub struct Relu;

impl FallibleModule for Relu {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(input.f_relu()?)
    }
}

impl Activation for Relu {}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::Tensor;

    use super::GeluNew;
    use crate::module::FallibleModule;

    #[test]
    fn gelu_new_returns_correct_values() {
        let gelu_new = GeluNew;
        let activations: ArrayD<f32> = (&gelu_new
            .forward(&Tensor::of_slice(&[-1., -0.5, 0., 0.5, 1.]))
            .unwrap())
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(
            activations,
            array![-0.1588, -0.1543, 0.0000, 0.3457, 0.8412].into_dyn(),
            epsilon = 1e-4
        );
    }
}
