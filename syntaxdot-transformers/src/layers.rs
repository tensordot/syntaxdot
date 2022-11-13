//! Basic neural network modules.
//!
//! These are modules that are not provided by the Torch binding, or where
//! different behavior is required from the modules.

use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::{ConvConfig, Init};
use tch::{self, Tensor};

use crate::module::{FallibleModule, FallibleModuleT};
use crate::TransformerError;

/// 1-D convolution.
#[derive(Debug)]
pub struct Conv1D {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
    pub config: ConvConfig,
}

impl Conv1D {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        in_features: i64,
        out_features: i64,
        kernel_size: i64,
        groups: i64,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        let config = ConvConfig {
            groups,
            ..ConvConfig::default()
        };

        let bs = if config.bias {
            Some(vs.var("bias", &[out_features], config.bs_init)?)
        } else {
            None
        };

        let ws = vs.var(
            "weight",
            &[out_features, in_features / groups, kernel_size],
            config.ws_init,
        )?;

        Ok(Conv1D { ws, bs, config })
    }
}

impl FallibleModule for Conv1D {
    type Error = TransformerError;

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(Tensor::f_conv1d(
            xs,
            &self.ws,
            self.bs.as_ref(),
            &[self.config.stride],
            &[self.config.padding],
            &[self.config.dilation],
            self.config.groups,
        )?)
    }
}

/// Dropout layer.
///
/// This layer zeros out random elements of a tensor with probability
/// *p*. Dropout is a form of regularization and prevents
/// co-adaptation of neurons.
#[derive(Debug)]
pub struct Dropout {
    p: f64,
}

impl Dropout {
    /// Drop out elements with probability *p*.
    pub fn new(p: f64) -> Self {
        Dropout { p }
    }
}

impl FallibleModuleT for Dropout {
    type Error = TransformerError;

    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        Ok(input.f_dropout(self.p, train)?)
    }
}

/// Embedding lookup layer.
#[derive(Debug)]
pub struct Embedding(pub Tensor);

impl Embedding {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        name: &str,
        num_embeddings: i64,
        embedding_dim: i64,
        init: Init,
    ) -> Result<Self, TransformerError> {
        Ok(Embedding(vs.borrow().var(
            name,
            &[num_embeddings, embedding_dim],
            init,
        )?))
    }
}

impl FallibleModule for Embedding {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(Tensor::f_embedding(&self.0, input, -1, false, false)?)
    }
}

/// Layer that applies layer normalization.
#[derive(Debug)]
pub struct LayerNorm {
    eps: f64,
    normalized_shape: Vec<i64>,

    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

impl LayerNorm {
    /// Construct a layer normalization layer.
    ///
    /// The mean and standard deviation are computed over the last
    /// number of dimensions with the shape defined by
    /// `normalized_shape`. If `elementwise_affine` is `True`, a
    /// learnable affine transformation of the shape
    /// `normalized_shape` is added after normalization.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        normalized_shape: impl Into<Vec<i64>>,
        eps: f64,
        elementwise_affine: bool,
    ) -> Self {
        let vs = vs.borrow();

        let normalized_shape = normalized_shape.into();

        let (weight, bias) = if elementwise_affine {
            (
                Some(vs.ones("weight", &normalized_shape)),
                Some(vs.zeros("bias", &normalized_shape)),
            )
        } else {
            (None, None)
        };

        LayerNorm {
            eps,
            normalized_shape,
            weight,
            bias,
        }
    }
}

impl FallibleModule for LayerNorm {
    type Error = TransformerError;

    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        // XXX: last parameter is `cudnn_enable`. What happens if we always
        //      set this to `true`?
        Ok(input.f_layer_norm(
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
            false,
        )?)
    }
}

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

    pub pairwise: bool,
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
    pairwise: bool,
}

impl PairwiseBilinear {
    /// Construct a new bilinear layer.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &PairwiseBilinearConfig,
    ) -> Result<Self, TransformerError> {
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
        )?;

        Ok(PairwiseBilinear {
            bias_u: config.bias_u,
            bias_v: config.bias_v,
            weight,
            pairwise: config.pairwise,
        })
    }

    /// Apply this layer to the given inputs.
    ///
    /// Both inputs must have the same shape. Returns a tensor of
    /// shape `[batch_size, seq_len, seq_len, out_features]` given
    /// inputs of shape `[batch_size, seq_len, in_features]`.
    pub fn forward(&self, u: &Tensor, v: &Tensor) -> Result<Tensor, TransformerError> {
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

        let (batch_size, seq_len, _) = u.size3()?;

        let ones = Tensor::ones(&[batch_size, seq_len, 1], (u.kind(), u.device()));

        let u = if self.bias_u {
            Tensor::f_cat(&[u, &ones], -1)?
        } else {
            u.shallow_clone()
        };

        let v = if self.bias_v {
            Tensor::f_cat(&[v, &ones], -1)?
        } else {
            v.shallow_clone()
        };

        if self.pairwise {
            // [batch_size, max_seq_len, out_features, v features].
            let intermediate = Tensor::f_einsum("blu,uov->blov", &[&u, &self.weight], None)?;

            // We perform a matrix multiplication to get the output with
            // the shape [batch_size, seq_len, seq_len, out_features].
            let bilinear = Tensor::f_einsum("bmv,blov->bmlo", &[&v, &intermediate], None)?;

            Ok(bilinear.f_squeeze_dim(-1)?)
        } else {
            Ok(Tensor::f_einsum(
                "blu,uov,blv->blo",
                &[&u, &self.weight, &v],
                None,
            )?)
        }
    }
}

/// Variational dropout (Gal and Ghahramani, 2016)
///
/// For a tensor with `[batch_size, seq_len, repr_size]`, apply
/// the same dropout `[batch_size, 1, repr_size]` to each sequence
/// element.
#[derive(Debug)]
pub struct VariationalDropout {
    p: f64,
}

impl VariationalDropout {
    /// Create a variational dropout layer with the given dropout probability.
    pub fn new(p: f64) -> Self {
        VariationalDropout { p }
    }
}

impl FallibleModuleT for VariationalDropout {
    type Error = TransformerError;

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        // Avoid unnecessary work during prediction.
        if !train {
            return Ok(xs.shallow_clone());
        }

        let (batch_size, _, repr_size) = xs.size3()?;
        let dropout_mask = Tensor::f_ones(&[batch_size, 1, repr_size], (xs.kind(), xs.device()))?
            .f_dropout_(self.p, true)?;
        Ok(xs.f_mul(&dropout_mask)?)
    }
}

#[cfg(test)]
mod tests {
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use syntaxdot_tch_ext::RootExt;

    use crate::layers::{PairwiseBilinear, PairwiseBilinearConfig};

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
                pairwise: true,
            },
        )
        .unwrap();

        assert_eq!(
            bilinear.forward(&input1, &input2).unwrap().size(),
            &[64, 10, 10, 5]
        );
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
                pairwise: true,
            },
        )
        .unwrap();

        assert_eq!(
            bilinear.forward(&input1, &input2).unwrap().size(),
            &[64, 10, 10]
        );
    }
}
