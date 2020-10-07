//! CoNLL-X layer encoder.

use std::convert::{Infallible, TryFrom};

use conllu::graph::{Node, Sentence};
use conllu::token::{Features, Token};
use serde_derive::{Deserialize, Serialize};

use super::{EncodingProb, SentenceDecoder, SentenceEncoder};

mod error;
use self::error::*;

/// Tagging layer.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Layer {
    UPos,
    XPos,

    /// A specific morphologic feature.
    Feature {
        feature: String,

        // Default value if the feature is absent.
        default: Option<String>,
    },

    /// All morphological features represented as a string.
    #[serde(rename = "feature_string")]
    FeatureString,

    Misc {
        feature: String,

        // Default value if the feature is absent.
        default: Option<String>,
    },
}

impl Layer {
    /// Construct a feature layer.
    pub fn feature(feature: String, default: Option<String>) -> Self {
        Layer::Feature { feature, default }
    }

    /// Construct a miscellaneous feature layer.
    pub fn misc(feature: String, default: Option<String>) -> Self {
        Layer::Misc { feature, default }
    }
}

/// Layer values.
pub trait LayerValue {
    /// Set a layer value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>);

    /// Get a layer value.
    fn value(&self, layer: &Layer) -> Option<String>;
}

impl LayerValue for Token {
    /// Set the layer to the given value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>) {
        let value = value.into();

        match layer {
            Layer::UPos => {
                self.set_upos(Some(value));
            }
            Layer::XPos => {
                self.set_xpos(Some(value));
            }
            Layer::Feature { feature, .. } => {
                self.features_mut().insert(feature.clone(), value);
            }
            Layer::FeatureString => {
                self.set_features(
                    Features::try_from(value.as_str()).expect("Invalid feature representation"),
                );
            }
            Layer::Misc { feature, .. } => {
                self.misc_mut().insert(feature.clone(), Some(value));
            }
        };
    }

    /// Look up the layer value in a token.
    fn value(&self, layer: &Layer) -> Option<String> {
        match layer {
            Layer::UPos => self.upos().map(ToOwned::to_owned),
            Layer::XPos => self.xpos().map(ToOwned::to_owned),
            Layer::FeatureString => Some(self.features().into()),
            Layer::Feature { feature, default } => self
                .features()
                .get(feature)
                .cloned()
                .or_else(|| default.clone()),
            Layer::Misc { feature, default } => match self.misc().get(feature) {
                // Feature with an associated value.
                Some(Some(ref val)) => Some(val.clone()),

                // Feature without an associated value, should not be used.
                Some(None) => None,

                // The feature is absent.
                None => default.clone(),
            },
        }
    }
}

/// Encode sentences using a CoNLL-X layer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LayerEncoder {
    layer: Layer,
}

impl LayerEncoder {
    /// Construct a new layer encoder of the given layer.
    pub fn new(layer: Layer) -> Self {
        LayerEncoder { layer }
    }
}

impl SentenceDecoder for LayerEncoder {
    type Encoding = String;

    type Error = Infallible;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        assert_eq!(
            labels.len(),
            sentence.len() - 1,
            "Labels and sentence length mismatch"
        );

        for (token, token_labels) in sentence
            .iter_mut()
            .filter_map(Node::token_mut)
            .zip(labels.iter())
        {
            if let Some(label) = token_labels.as_ref().get(0) {
                token.set_value(&self.layer, label.encoding().as_str());
            }
        }

        Ok(())
    }
}

impl SentenceEncoder for LayerEncoder {
    type Encoding = String;

    type Error = EncodeError;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);
        for token in sentence.iter().filter_map(Node::token) {
            let label = token
                .value(&self.layer)
                .ok_or_else(|| EncodeError::MissingLabel {
                    form: token.form().to_owned(),
                })?;
            encoding.push(label.to_owned());
        }

        Ok(encoding)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use conllu::token::{Features, Misc, Token, TokenBuilder};

    use crate::layer::{Layer, LayerValue};

    #[test]
    fn layer() {
        let token: Token = TokenBuilder::new("test")
            .upos("CP")
            .xpos("P")
            .features(Features::try_from("c=d|a=b").unwrap())
            .misc(Misc::from("u=v|x=y"))
            .into();

        assert_eq!(token.value(&Layer::UPos), Some("CP".to_string()));
        assert_eq!(token.value(&Layer::XPos), Some("P".to_string()));
        assert_eq!(
            token.value(&Layer::feature("a".to_owned(), None)),
            Some("b".to_string())
        );
        assert_eq!(
            token.value(&Layer::feature("c".to_owned(), None)),
            Some("d".to_string())
        );
        assert_eq!(token.value(&Layer::feature("e".to_owned(), None)), None);
        assert_eq!(
            token.value(&Layer::feature(
                "e".to_owned(),
                Some("some_default".to_string())
            )),
            Some("some_default".to_string())
        );
        assert_eq!(
            token.value(&Layer::FeatureString),
            Some("a=b|c=d".to_string())
        );

        assert_eq!(
            token.value(&Layer::misc("u".to_owned(), None)),
            Some("v".to_string())
        );
        assert_eq!(
            token.value(&Layer::misc("x".to_owned(), None)),
            Some("y".to_string())
        );
        assert_eq!(token.value(&Layer::misc("z".to_owned(), None)), None);
        assert_eq!(
            token.value(&Layer::misc(
                "z".to_owned(),
                Some("some_default".to_string())
            )),
            Some("some_default".to_string())
        );
    }

    #[test]
    fn set_layer() {
        let mut token: Token = TokenBuilder::new("test").into();

        assert_eq!(token.value(&Layer::FeatureString), Some("_".to_string()));

        token.set_value(&Layer::UPos, "CP");
        token.set_value(&Layer::XPos, "P");
        token.set_value(&Layer::feature("a".to_owned(), None), "b");
        token.set_value(&Layer::misc("u".to_owned(), None), "v");

        assert_eq!(token.value(&Layer::UPos), Some("CP".to_string()));
        assert_eq!(token.value(&Layer::XPos), Some("P".to_string()));
        assert_eq!(
            token.value(&Layer::feature("a".to_owned(), None)),
            Some("b".to_string())
        );
        assert_eq!(token.value(&Layer::feature("c".to_owned(), None)), None);
        assert_eq!(token.value(&Layer::FeatureString), Some("a=b".to_string()));

        assert_eq!(
            token.value(&Layer::misc("u".to_owned(), None)),
            Some("v".to_string())
        );
        assert_eq!(token.value(&Layer::misc("x".to_owned(), None)), None);
    }
}
