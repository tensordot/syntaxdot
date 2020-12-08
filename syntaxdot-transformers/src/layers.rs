use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::{ConvConfig, Init, Linear, Module, ModuleT};
use tch::{self, Tensor};

/// Trait to place layer tensors in the var store.
pub trait PlaceInVarStore
where
    Self: Sized,
{
    /// Place layer tensors in the var store.
    ///
    /// This method replaces a layer's tensors by tensors that are
    /// in the given var store.
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<PathExt<'a>>);

    /// Place layer tensors in the var store.
    fn place_in_var_store<'a>(mut self, vs: impl Borrow<PathExt<'a>>) -> Self {
        self.place_in_var_store_inplace(vs);
        self
    }
}

impl PlaceInVarStore for Linear {
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<PathExt<'a>>) {
        let vs = vs.borrow();
        self.ws = vs.var_copy("weight", &self.ws);
        self.bs = vs.var_copy("bias", &self.bs);
    }
}

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
    ) -> Self {
        let vs = vs.borrow();

        let config = ConvConfig {
            groups,
            ..ConvConfig::default()
        };

        let bs = if config.bias {
            Some(vs.var("bias", &[out_features], config.bs_init))
        } else {
            None
        };

        let ws = vs.var(
            "weight",
            &[out_features, in_features / groups, kernel_size],
            config.ws_init,
        );

        Conv1D { ws, bs, config }
    }
}

impl Module for Conv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv1d(
            xs,
            &self.ws,
            self.bs.as_ref(),
            &[self.config.stride],
            &[self.config.padding],
            &[self.config.dilation],
            self.config.groups,
        )
    }
}

impl PlaceInVarStore for Conv1D {
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<PathExt<'a>>) {
        let vs = vs.borrow();
        self.ws = vs.var_copy("weight", &self.ws);
        self.bs = self.bs.as_ref().map(|bs| vs.var_copy("bias", bs));
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
    ) -> Self {
        Embedding(
            vs.borrow()
                .var(name, &[num_embeddings, embedding_dim], init),
        )
    }
}

impl PlaceInVarStore for Embedding {
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<PathExt<'a>>) {
        self.0 = vs.borrow().var_copy("embeddings", &self.0)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        Tensor::embedding(&self.0, input, -1, false, false)
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        input.dropout(self.p, train)
    }
}

/// Layer that applies layer normalization.
#[derive(Debug)]
pub struct LayerNorm {
    elementwise_affine: bool,
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
            elementwise_affine,
            normalized_shape,

            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        // XXX: last parameter is `cudnn_enable`. What happens if we always
        //      set this to `true`?
        input.layer_norm(
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
            false,
        )
    }
}

impl PlaceInVarStore for LayerNorm {
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<PathExt<'a>>) {
        let vs = vs.borrow();

        self.weight = self
            .weight
            .as_ref()
            .map(|weight| vs.var_copy("weight", weight));
        self.bias = self.bias.as_ref().map(|bias| vs.var_copy("bias", bias));
    }
}
