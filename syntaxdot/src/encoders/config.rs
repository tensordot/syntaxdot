use std::ops::Deref;

use serde::{Deserialize, Serialize};
use syntaxdot_encoders::layer::Layer;
use syntaxdot_encoders::lemma::BackoffStrategy;

/// Configuration of a set of encoders.
///
/// The configuration is a mapping from encoder name to
/// encoder configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodersConfig(pub Vec<NamedEncoderConfig>);

impl Deref for EncodersConfig {
    type Target = [NamedEncoderConfig];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The type of encoder.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EncoderType {
    /// Lemma encoder using edit trees.
    Lemma(BackoffStrategy),

    /// Encoder for plain sequence labels.
    Sequence(Layer),

    /// Lemma encoder using edit trees, with TüBa-D/Z-specific
    /// transformations.
    TdzLemma(BackoffStrategy),
}

/// Configuration of an encoder with a name.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamedEncoderConfig {
    pub encoder: EncoderType,
    pub name: String,
}
