use thiserror::Error;

/// Transformer errors.
#[derive(Clone, Debug, Error)]
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

    /// The activation function is unknown.
    #[error("unknown activation function: {activation:?}")]
    UnknownActivationFunction { activation: String },
}

impl TransformerError {
    pub(crate) fn unknown_activation_function(activation: impl Into<String>) -> Self {
        TransformerError::UnknownActivationFunction {
            activation: activation.into(),
        }
    }
}
