use std::io;

use ndarray::ShapeError;
use syntaxdot_encoders::dependency;
use syntaxdot_tokenizers::TokenizerError;
use syntaxdot_transformers::TransformerError;
use tch::TchError;
use thiserror::Error;

use crate::encoders::{DecoderError, EncoderError};

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum SyntaxDotError {
    #[error(transparent)]
    BertError(#[from] TransformerError),

    #[allow(clippy::upper_case_acronyms)]
    #[error(transparent)]
    ConlluIOError(#[from] conllu::IOError),

    #[error(transparent)]
    DecoderError(#[from] DecoderError),

    #[error(transparent)]
    DependencyEncodeError(#[from] dependency::EncodeError),

    #[error(transparent)]
    EncoderError(#[from] EncoderError),

    #[error("Illegal configuration: {0}")]
    IllegalConfigurationError(String),

    #[allow(clippy::upper_case_acronyms)]
    #[error(transparent)]
    IOError(#[from] io::Error),

    #[error("The optimizer does not have any associated trainable variables")]
    NoTrainableVariables,

    #[error("{0}: {1}")]
    JSonSerialization(String, serde_json::Error),

    #[error("Cannot relativize path: {0}")]
    RelativizePathError(String),

    #[error(transparent)]
    ShapeError(#[from] ShapeError),

    #[error(transparent)]
    Tch(#[from] TchError),

    #[error(transparent)]
    #[allow(clippy::upper_case_acronyms)]
    TOMLDeserializationError(#[from] toml::de::Error),

    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),
}
