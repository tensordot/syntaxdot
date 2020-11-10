//! SqueezeBERT (Iandola et al., 2020)
//!
//! SqueezeBERT follows the same architecture as BERT, but replaces most
//! matrix multiplications by grouped convolutions. This reduces the
//! number of parameters and speeds up inference.

mod config;
pub use config::SqueezeBertConfig;

mod embeddings;

mod encoder;
pub use encoder::SqueezeBertEncoder;

mod layer;
pub(crate) use layer::SqueezeBertLayer;
