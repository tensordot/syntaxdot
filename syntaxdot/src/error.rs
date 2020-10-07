use std::io;

use ndarray::ShapeError;
use syntaxdot_transformers::models::bert::BertError;
use thiserror::Error;

use crate::encoders::{DecoderError, EncoderError};

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum SyntaxDotError {
    #[error(transparent)]
    BertError(#[from] BertError),

    #[error(transparent)]
    ConlluIOError(#[from] conllu::IOError),

    #[error(transparent)]
    DecoderError(#[from] DecoderError),

    #[error(transparent)]
    EncoderError(#[from] EncoderError),

    #[cfg(feature = "load-hdf5")]
    #[error(transparent)]
    HDF5Error(#[from] hdf5::Error),

    #[error(transparent)]
    IOError(#[from] io::Error),

    #[error("{0}: {1}")]
    JSonSerialization(String, serde_json::Error),

    #[error("Cannot relativize path: {0}")]
    RelativizePathError(String),

    #[error(transparent)]
    SentencePieceError(#[from] sentencepiece::SentencePieceError),

    #[error(transparent)]
    ShapeError(#[from] ShapeError),

    #[error(transparent)]
    TOMLDeserializationError(#[from] toml::de::Error),

    #[error(transparent)]
    WordPiecesError(#[from] wordpieces::WordPiecesError),
}
