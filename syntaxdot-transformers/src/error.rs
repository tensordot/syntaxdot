use tch::TchError;
use thiserror::Error;

/// Transformer errors.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransformerError {
    /// The hidden size is not a multiple of the number of attention heads.
    #[error("hidden size ({hidden_size:?}) is not a multiple of attention heads ({num_attention_heads:?})")]
    IncorrectHiddenSize {
        /// The hidden size.
        hidden_size: i64,

        /// The number of attention heads.
        num_attention_heads: i64,
    },

    /// Torch error.
    #[error(transparent)]
    Tch(#[from] TchError),

    /// The activation function is unknown.
    #[error("unknown activation function: {activation:?}")]
    UnknownActivationFunction { activation: String },
}
