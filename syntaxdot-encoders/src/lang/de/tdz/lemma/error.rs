use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LemmatizationError {
    #[error(transparent)]
    IO(#[from] io::Error),

    #[error(transparent)]
    Fst(#[from] fst::Error),
}
