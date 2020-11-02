//! Transformer models.

pub mod albert;

pub mod bert;
pub use bert::{BertConfig, BertEmbeddings, BertEncoder};

mod encoder;
pub use encoder::Encoder;

#[cfg(feature = "load-hdf5")]
#[cfg(feature = "model-tests")]
#[cfg(test)]
pub(crate) mod resources;

pub mod roberta;
pub use roberta::RobertaEmbeddings;

pub mod sinusoidal;
pub use sinusoidal::SinusoidalEmbeddings;

pub mod squeeze_albert;

pub mod squeeze_bert;

pub mod traits;
