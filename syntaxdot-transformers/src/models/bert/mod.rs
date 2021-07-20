//! BERT (Devlin et al., 2018)

mod config;
pub use config::BertConfig;

mod embeddings;
pub use embeddings::BertEmbeddings;

mod encoder;
pub use encoder::BertEncoder;

mod layer;
pub(crate) use layer::bert_linear;
pub use layer::BertLayer;
