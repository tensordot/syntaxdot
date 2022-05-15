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

    #[error(transparent)]
    ConlluError(#[from] conllu::Error),

    #[error(transparent)]
    DecoderError(#[from] DecoderError),

    #[error(transparent)]
    DependencyEncodeError(#[from] dependency::EncodeError),

    #[error(transparent)]
    EncoderError(#[from] EncoderError),

    #[error("Illegal configuration: {0}")]
    IllegalConfigurationError(String),

    #[error(transparent)]
    IoError(#[from] io::Error),

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
    TomlDeserializationError(#[from] toml::de::Error),

    #[error(transparent)]
    TokenizerError(#[from] TokenizerError),

    #[error(transparent)]
    UdgraphError(#[from] udgraph::Error),
}
