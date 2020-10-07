use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EncodeError {
    /// The token does not have a label.
    #[error("token without a label: '{form:?}'")]
    MissingLabel { form: String },
}
