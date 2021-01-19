use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::Init;
use tch::Tensor;

/// Configuration for the `Bilinear` layer.
#[derive(Clone, Copy, Debug)]
pub struct PairwiseBilinearConfig {
    /// The number of input features.
    pub in_features: i64,

    /// The number of output features.
    pub out_features: i64,

    /// Standard deviation for random initialization.
    pub initializer_range: f64,

    pub bias_u: bool,

    pub bias_v: bool,
}

/// Pairwise bilinear forms.
///
/// Given two batches with sequence length *n*, apply pairwise
/// bilinear forms to each timestep within a sequence.
#[derive(Debug)]
pub struct PairwiseBilinear {
    weight: Tensor,
    bias_u: bool,
    bias_v: bool,
}

impl PairwiseBilinear {
    /// Construct a new bilinear layer.
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &PairwiseBilinearConfig) -> Self {
        assert!(
            config.in_features > 0,
            "in_features should be > 0, was: {}",
            config.in_features,
        );

        assert!(
            config.out_features > 0,
            "out_features should be > 0, was: {}",
            config.out_features,
        );

        let vs = vs.borrow();

        let bias_u_dim = if config.bias_u { 1 } else { 0 };
        let bias_v_dim = if config.bias_v { 1 } else { 0 };

        // We would normally use a separate variable for biases to enable
        // treating biases using a different parameter group. However, in
        // this case, the bias is not a 'constant' scalar, vector, or matrix,
        // but actually interacts with the inputs. Therefore, it seems  more
        // appropriate to treat biases in a biaffine classifier as regular
        // trainable variables.
        let weight = vs.var(
            "weight",
            &[
                config.in_features + bias_u_dim,
                config.out_features,
                config.in_features + bias_v_dim,
            ],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        );

        PairwiseBilinear {
            bias_u: config.bias_u,
            bias_v: config.bias_v,
            weight,
        }
    }

    /// Apply this layer to the given inputs.
    ///
    /// Both inputs must have the same shape. Returns a tensor of
    /// shape `[batch_size, seq_len, seq_len, out_features]` given
    /// inputs of shape `[batch_size, seq_len, in_features]`.
    pub fn forward(&self, u: &Tensor, v: &Tensor) -> Tensor {
        assert_eq!(
            u.size(),
            v.size(),
            "Inputs to Bilinear must have the same shape: {:?} {:?}",
            u.size(),
            v.size()
        );

        assert_eq!(
            u.dim(),
            3,
            "Shape should have 3 dimensions, has: {}",
            u.dim()
        );

        let (batch_size, seq_len, _) = u.size3().unwrap();

        let ones = Tensor::ones(&[batch_size, seq_len, 1], (u.kind(), u.device()));

        let u = if self.bias_u {
            Tensor::cat(&[u, &ones], -1)
        } else {
            u.shallow_clone()
        };

        let v = if self.bias_v {
            Tensor::cat(&[v, &ones], -1)
        } else {
            v.shallow_clone()
        };

        // [batch_size, max_seq_len, out_features, v features].
        let intermediate = Tensor::einsum("blu,uov->blov", &[&u, &self.weight]);

        // We perform a matrix multiplication to get the output with
        // the shape [batch_size, seq_len, seq_len, out_features].
        let bilinear = Tensor::einsum("bmv,blov->bmlo", &[&v, &intermediate]);

        bilinear.squeeze1(-1)
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use syntaxdot_tch_ext::RootExt;

    use crate::model::pairwise_bilinear::{PairwiseBilinear, PairwiseBilinearConfig};

    #[test]
    fn bilinear_correct_shapes() {
        // Apply a bilinear layer to ensure that the shapes are correct.

        let input1 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));
        let input2 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(Device::Cpu);
        let bilinear = PairwiseBilinear::new(
            vs.root_ext(|_| 0),
            &PairwiseBilinearConfig {
                bias_u: true,
                bias_v: false,
                in_features: 200,
                out_features: 5,
                initializer_range: 0.02,
            },
        );

        assert_eq!(bilinear.forward(&input1, &input2).size(), &[64, 10, 10, 5]);
    }

    #[test]
    fn bilinear_1_output_correct_shapes() {
        let input1 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));
        let input2 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(Device::Cpu);
        let bilinear = PairwiseBilinear::new(
            vs.root_ext(|_| 0),
            &PairwiseBilinearConfig {
                bias_u: true,
                bias_v: false,
                in_features: 200,
                out_features: 1,
                initializer_range: 0.02,
            },
        );

        assert_eq!(bilinear.forward(&input1, &input2).size(), &[64, 10, 10]);
    }
}
