//! Activation functions

use std::convert::TryFrom;
use std::f64;

use serde::Deserialize;
use tch::Tensor;

use crate::module::FallibleModule;
use crate::TransformerError;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(try_from = "String")]
pub enum Activation {
    /// GELU activation function.
    ///
    /// GELU(x)=x Φ(x)
    ///
    /// where Φ(x) is the CDF for the Gaussian distribution.
    Gelu,

    /// GELU activation function (Google/OpenAI flavor).
    ///
    /// GELU(x)=x Φ(x)
    ///
    /// where Φ(x) is the CDF for the Gaussian distribution.
    GeluNew,

    /// ReLU activation function
    ///
    /// ReLU(x)=max(0,x)
    Relu,
}

impl TryFrom<&str> for Activation {
    type Error = TransformerError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "gelu" => Ok(Activation::Gelu),
            "gelu_new" => Ok(Activation::GeluNew),
            "relu" => Ok(Activation::Relu),
            unknown => Err(TransformerError::UnknownActivationFunction {
                activation: unknown.to_string(),
            }),
        }
    }
}

impl TryFrom<String> for Activation {
    type Error = TransformerError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::try_from(value.as_str())
    }
}

impl FallibleModule for Activation {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        match self {
            Self::Gelu => Ok(input.f_gelu()?),
            Self::GeluNew => Ok(0.5
                * input
                * (1.0
                    + Tensor::f_tanh(
                        &((2. / f64::consts::PI).sqrt()
                            * (input + 0.044715 * input.pow_tensor_scalar(3.0))),
                    )?)),
            Self::Relu => Ok(input.f_relu()?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::Tensor;

    use crate::activations::Activation;
    use crate::module::FallibleModule;

    #[test]
    fn gelu_new_returns_correct_values() {
        let gelu_new = Activation::GeluNew;
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
