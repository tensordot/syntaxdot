//! ALBERT (Lan et al., 2020)

mod config;
pub use config::AlbertConfig;

mod embeddings;
pub(crate) use embeddings::AlbertEmbeddingProjection;
pub use embeddings::AlbertEmbeddings;

mod encoder;
pub use encoder::AlbertEncoder;
