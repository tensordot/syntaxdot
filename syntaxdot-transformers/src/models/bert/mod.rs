//! BERT (Devlin et al., 2018)

mod config;
pub use config::BertConfig;

mod embeddings;
pub use embeddings::BertEmbeddings;

mod encoder;
pub use encoder::BertEncoder;

mod layer;
pub use layer::BertLayer;
pub(crate) use layer::{bert_activations, bert_linear};
