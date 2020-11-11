use thiserror::Error;

mod encoder;
pub use self::encoder::{BackoffStrategy, EditTreeEncoder};

mod edit_tree;
pub use edit_tree::EditTree;

mod tdz;
pub use tdz::TdzLemmaEncoder;

/// Lemma encoding error.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EncodeError {
    /// The token does not have a lemma.
    #[error("token without a lemma: '{form:?}'")]
    MissingLemma { form: String },

    /// No edit tree can be constructed.
    #[error("cannot find an edit tree that rewrites '{form:?}' into '{lemma:?}'")]
    NoEditTree { form: String, lemma: String },
}
