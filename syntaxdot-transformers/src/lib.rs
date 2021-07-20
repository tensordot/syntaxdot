//! Transformer models (Vaswani et al., 2017)
//!
//! This crate implements various transformer models, provided through
//! the [`models`] module. The implementations are more restricted than
//! e.g. their Huggingface counterparts, focusing only on the parts
//! necessary for sequence labeling.

pub mod activations;

pub(crate) mod cow;

mod error;
pub use error::TransformerError;

pub mod layers;

pub mod loss;

pub mod models;

pub mod module;

pub mod scalar_weighting;

pub(crate) mod util;
