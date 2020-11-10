use std::io;

use sentencepiece::SentencePieceError;
use thiserror::Error;
use wordpieces::WordPiecesError;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Cannot open tokenizer model `{model_path:?}`: {inner:?}")]
    OpenError {
        model_path: String,
        inner: io::Error,
    },

    #[error(transparent)]
    SentencePiece(#[from] SentencePieceError),

    #[error("Cannot process word pieces: {0}")]
    WordPieces(#[from] WordPiecesError),
}

impl TokenizerError {
    pub fn open_error(model_path: impl Into<String>, inner: io::Error) -> Self {
        TokenizerError::OpenError {
            model_path: model_path.into(),
            inner,
        }
    }
}
