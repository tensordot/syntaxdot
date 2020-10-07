use std::ops::Deref;

use tch::{Device, Kind, Tensor};

/// Mask of logit values
///
/// This mask masks logits by setting inactive logits to a
/// large negative value (`-10_000`).
pub struct LogitsMask {
    inner: Tensor,
}

impl LogitsMask {
    /// Construct a logits mask from a boolean mask (consisting of
    /// bools, or a numeric type with 0/1).
    pub fn from_bool_mask(mask: &Tensor) -> Self {
        assert_eq!(
            mask.size().len(),
            2,
            "Expected a mask of shape [batch_size, timesteps]"
        );

        // The attention mask has shape [batch_size, seq_len], extend
        // to [batch_size, 1, 1, seq_len].
        let extended_mask = mask.unsqueeze(1).unsqueeze(1);

        // Use (very) negative values for time steps that should be masked.
        let logits_mask = (1.0 - extended_mask.to_kind(Kind::Float)) * -10_000.;

        LogitsMask { inner: logits_mask }
    }
}

impl Deref for LogitsMask {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Trait for realizing sinusoidal positions into a tensor.
pub trait SinusoidalPositions {
    /// Create sinusoidal positions in-place.
    ///
    /// If `p_norm` is specified, the sinusoidal embeddings are
    /// normalized using their *p* norm. For instance using *p = 2*
    /// will result in embeddings that are unit vectors.
    ///
    /// This method panics if the the shape of tensor is not `[a, b]`,
    /// where `b % 2 == 0`.
    fn sinusoidal_positions_(&mut self, p_norm: Option<f64>);

    /// Create new tensor with sinusoidal positions.
    ///
    /// The number of dimensions should be even.
    fn sinusoidal_positions(
        n_positions: i64,
        dims: i64,
        p_norm: Option<f64>,
        options: (Kind, Device),
    ) -> Self;
}

impl SinusoidalPositions for Tensor {
    fn sinusoidal_positions_(&mut self, p_norm: Option<f64>) {
        let shape = self.size();
        let dims = shape[1];

        assert_eq!(
            shape.len(),
            2,
            "Sinusoidal positions should be realized into a matrix"
        );

        assert!(
            dims % 2 == 0,
            "Dimensionality of sinusoidal positions should be even, was: {}",
            dims
        );

        let self_shape = self.size();
        let num_embeddings = self_shape[0];
        let embedding_dim = self_shape[1];

        // Vaswani et al, 2017:
        //
        // let x = 2i, then
        // PE(pos, x) = sin(pos / 10000^(x/d))
        // PE(pos, x + 1) = cos(pos / 10000^(x/d))
        //
        //   pos / 10000^(x/d)
        // = pos * (1 / 10000^(x/d))
        // = pos * exp(ln(1) - ln(10000^(x/d)))
        // = pos * exp(-ln(10000) (x/d))
        // = pos * exp(x * (-ln(10000) / d))
        //
        // Avoids the use of larger numbers with decreased precision.
        let position = Tensor::arange(num_embeddings, (Kind::Float, self.device())).unsqueeze(1);
        let div_term = (Tensor::arange2(0, embedding_dim, 2, (Kind::Float, self.device()))
            * (-(10_000f64.ln()) / embedding_dim as f64))
            .exp();
        let position_encodings = position * div_term;

        // Copy the sinusoidal embeddings into the output shape. Run with
        // no_grad to ensure that the tensors created in this function do
        // not become leaf nodes of the graph.
        tch::no_grad(|| {
            self.slice(1, 0, embedding_dim, 2)
                .copy_(&position_encodings.sin());
            self.slice(1, 1, embedding_dim, 2)
                .copy_(&position_encodings.cos());

            if let Some(p) = p_norm {
                // Compute the p-norm.
                let norm = self.norm2(p, &[-1], true);

                // Normalize embeddings.
                *self /= &norm;
            }
        });
    }

    fn sinusoidal_positions(
        n_positions: i64,
        dims: i64,
        p_norm: Option<f64>,
        options: (Kind, Device),
    ) -> Self {
        assert!(
            dims % 2 == 0,
            "Dimensionality of sinusoidal positions should be even, was: {}",
            dims
        );

        let mut positions = Tensor::empty(&[n_positions, dims], options);
        positions.sinusoidal_positions_(p_norm);

        positions
    }
}

#[cfg(test)]
pub mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::{Device, Kind, Tensor};

    use crate::util::SinusoidalPositions;

    #[test]
    #[should_panic]
    fn positions_dimensionality_must_be_even() {
        let _positions: Tensor =
            SinusoidalPositions::sinusoidal_positions(5, 9, None, (Kind::Float, Device::Cpu));
    }

    #[test]
    fn positions_are_l1_normalized() {
        let positions: Tensor =
            SinusoidalPositions::sinusoidal_positions(5, 8, Some(1.), (Kind::Float, Device::Cpu));
        let norms: ArrayD<f32> = (&positions.abs().sum1(&[-1], false, Kind::Float))
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(norms, array![1., 1., 1., 1., 1.].into_dyn(), epsilon = 1e-4);
    }

    #[test]
    fn positions_are_l2_normalized() {
        let positions: Tensor =
            SinusoidalPositions::sinusoidal_positions(5, 8, Some(2.), (Kind::Float, Device::Cpu));
        let norms: ArrayD<f32> = (&positions.norm2(2., &[-1], false)).try_into().unwrap();
        assert_abs_diff_eq!(norms, array![1., 1., 1., 1., 1.].into_dyn(), epsilon = 1e-4);
    }

    #[test]
    fn positions_are_sinusoidal() {
        let positions: Tensor =
            SinusoidalPositions::sinusoidal_positions(5, 8, None, (Kind::Float, Device::Cpu));

        let positions: ArrayD<f32> = (&positions).try_into().unwrap();

        assert_abs_diff_eq!(
            positions,
            array![
                [
                    0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00,
                    0.0000e+00, 1.0000e+00
                ],
                [
                    8.4147e-01, 5.4030e-01, 9.9833e-02, 9.9500e-01, 9.9998e-03, 9.9995e-01,
                    1.0000e-03, 1.0000e+00
                ],
                [
                    9.0930e-01,
                    -4.1615e-01,
                    1.9867e-01,
                    9.8007e-01,
                    1.9999e-02,
                    9.9980e-01,
                    2.0000e-03,
                    1.0000e+00
                ],
                [
                    1.4112e-01,
                    -9.8999e-01,
                    2.9552e-01,
                    9.5534e-01,
                    2.9996e-02,
                    9.9955e-01,
                    3.0000e-03,
                    1.0000e+00
                ],
                [
                    -7.5680e-01,
                    -6.5364e-01,
                    3.8942e-01,
                    9.2106e-01,
                    3.9989e-02,
                    9.9920e-01,
                    4.0000e-03,
                    9.9999e-01
                ]
            ]
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    #[should_panic]
    fn positions_tensor_must_be_matrix() {
        let mut positions = Tensor::empty(&[8, 8, 8], (Kind::Float, Device::Cpu));
        positions.sinusoidal_positions_(None);
    }
}
