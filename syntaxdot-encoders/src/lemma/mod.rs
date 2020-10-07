use thiserror::Error;

mod edit_tree;
pub use self::edit_tree::{BackoffStrategy, EditTreeEncoder};

mod tdz;
pub use tdz::TdzLemmaEncoder;

/// Lemma encoding error.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EncodeError {
    /// The token does not have a lemma.
    #[error("token without a lemma: '{form:?}'")]
    MissingLemma { form: String },
}
