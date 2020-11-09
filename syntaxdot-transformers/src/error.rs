use thiserror::Error;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TransformerError {
    #[cfg(feature = "load-hdf5")]
    #[error(transparent)]
    HDF5(#[from] hdf5::Error),

    #[error("hidden size ({hidden_size:?}) is not a multiple of attention heads ({num_attention_heads:?})")]
    IncorrectHiddenSize {
        hidden_size: i64,
        num_attention_heads: i64,
    },

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
