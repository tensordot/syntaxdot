use std::borrow::Borrow;

use tch::nn::{Init, Path};
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
}

/// Pairwise bilinear forms.
///
/// Given two batches with sequence length *n*, apply pairwise
/// bilinear forms to each timestep within a sequence.
#[derive(Debug)]
pub struct PairwiseBilinear {
    weight: Tensor,
}

impl PairwiseBilinear {
    /// Construct a new bilinear layer.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &PairwiseBilinearConfig) -> Self {
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

        let weight = vs.var(
            "weight",
            &[config.in_features, config.out_features, config.in_features],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        );

        PairwiseBilinear { weight }
    }

    /// Apply this layer to the given inputs.
    ///
    /// Both inputs must have the same shape. Returns a tensor of
    /// shape `[batch_size, seq_len, seq_len, out_features]` given
    /// inputs of shape `[batch_size, seq_len, in_features]`.
    pub fn forward(&self, input1: &Tensor, input2: &Tensor) -> Tensor {
        assert_eq!(
            input1.size(),
            input2.size(),
            "Inputs to Bilinear must have the same shape: {:?} {:?}",
            input1.size(),
            input2.size()
        );

        assert_eq!(
            input1.dim(),
            3,
            "Shape should have 3 dimensions, has: {}",
            input1.dim()
        );

        // The shapes of the inputs are [batch_size, max_seq_len, features].
        // After matrix multiplication, we get the intermediate shape
        // [batch_size, max_seq_len, out_features, in_features].
        let intermediate = Tensor::einsum("blh,hfh->blfh", &[input1, &self.weight]);

        // We perform a matrix multiplication to get the output with
        // the shape [batch_size, seq_len, seq_len, out_features].
        Tensor::einsum("blh,bmfh->blmf", &[input2, &intermediate])
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use crate::model::pairwise_bilinear::{PairwiseBilinear, PairwiseBilinearConfig};

    #[test]
    fn bilinear_correct_shapes() {
        // Apply a bilinear layer to ensure that the shapes are correct.

        let input1 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));
        let input2 = Tensor::rand(&[64, 10, 200], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(Device::Cpu);
        let bilinear = PairwiseBilinear::new(
            vs.root(),
            &PairwiseBilinearConfig {
                in_features: 200,
                out_features: 5,
                initializer_range: 0.02,
            },
        );

        assert_eq!(bilinear.forward(&input1, &input2).size(), &[64, 10, 10, 5]);
    }
}
