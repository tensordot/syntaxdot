use std::ops::Deref;

use serde::{Deserialize, Serialize};
use syntaxdot_encoders::depseq::PosLayer;
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
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncoderType {
    /// Encoder for syntactical dependencies.
    Dependency {
        encoder: DependencyEncoder,
        root_relation: String,
    },

    /// Lemma encoder using edit trees.
    Lemma(BackoffStrategy),

    /// Encoder for plain sequence labels.
    Sequence(Layer),

    /// Lemma encoder using edit trees, with TüBa-D/Z-specific
    /// transformations.
    TdzLemma(BackoffStrategy),
}

/// The type of dependency encoder.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DependencyEncoder {
    /// Encode a token's head by relative position.
    RelativePosition,

    /// Encode a token's head by relative position of the POS tag.
    RelativePos(PosLayer),
}

/// Configuration of an encoder with a name.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamedEncoderConfig {
    pub encoder: EncoderType,
    pub name: String,
}
