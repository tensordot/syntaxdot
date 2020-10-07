use std::f64;

use tch::nn::Module;
use tch::Tensor;

pub trait Activation: Clone + Module {}

#[derive(Clone, Copy, Debug)]
pub struct GELUNew;

impl Module for GELUNew {
    fn forward(&self, input: &Tensor) -> Tensor {
        0.5 * input
            * (1.0
                + Tensor::tanh(
                    &((2. / f64::consts::PI).sqrt() * (input + 0.044715 * input.pow(3.0))),
                ))
    }
}

impl Activation for GELUNew {}

#[derive(Clone, Copy, Debug)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
    }
}

impl Activation for GELU {}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::nn::Module;
    use tch::Tensor;

    use super::GELUNew;

    #[test]
    fn gelu_new_returns_correct_values() {
        let gelu_new = GELUNew;
        let activations: ArrayD<f32> = (&gelu_new
            .forward(&Tensor::of_slice(&[-1., -0.5, 0., 0.5, 1.])))
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(
            activations,
            array![-0.1588, -0.1543, 0.0000, 0.3457, 0.8412].into_dyn(),
            epsilon = 1e-4
        );
    }
}
