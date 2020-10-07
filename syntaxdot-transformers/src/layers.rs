use std::borrow::Borrow;

use tch::nn::{Init, Linear, Module, ModuleT, Path};
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
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<Path<'a>>);

    /// Place layer tensors in the var store.
    fn place_in_var_store<'a>(mut self, vs: impl Borrow<Path<'a>>) -> Self {
        self.place_in_var_store_inplace(vs);
        self
    }
}

impl PlaceInVarStore for Linear {
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<Path<'a>>) {
        let vs = vs.borrow();
        self.ws = vs.var_copy("weight", &self.ws);
        self.bs = vs.var_copy("bias", &self.bs);
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
        vs: impl Borrow<Path<'a>>,
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
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<Path<'a>>) {
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
    #[cfg(feature = "load-hdf5")]
    pub(crate) fn new_with_affine(
        normalized_shape: impl Into<Vec<i64>>,
        eps: f64,
        weight: Tensor,
        bias: Tensor,
    ) -> Self {
        let normalized_shape = normalized_shape.into();

        LayerNorm {
            eps,
            elementwise_affine: true,
            normalized_shape,

            weight: Some(weight),
            bias: Some(bias),
        }
    }

    /// Construct a layer normalization layer.
    ///
    /// The mean and standard deviation are computed over the last
    /// number of dimensions with the shape defined by
    /// `normalized_shape`. If `elementwise_affine` is `True`, a
    /// learnable affine transformation of the shape
    /// `normalized_shape` is added after normalization.
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
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
    fn place_in_var_store_inplace<'a>(&mut self, vs: impl Borrow<Path<'a>>) {
        let vs = vs.borrow();

        self.weight = self
            .weight
            .as_ref()
            .map(|weight| vs.var_copy("weight", weight));
        self.bias = self.bias.as_ref().map(|bias| vs.var_copy("bias", bias));
    }
}
