use std::hash::Hash;
use std::ops::Deref;

use numberer::Numberer;
use serde::{Deserialize, Serialize};
use syntaxdot_encoders::categorical::{ImmutableCategoricalEncoder, MutableCategoricalEncoder};
use syntaxdot_encoders::depseq::{
    DependencyEncoding, RelativePos, RelativePosEncoder, RelativePosition, RelativePositionEncoder,
};
use syntaxdot_encoders::layer::LayerEncoder;
use syntaxdot_encoders::lemma::{EditTree, EditTreeEncoder, TdzLemmaEncoder};
use syntaxdot_encoders::{EncodingProb, SentenceDecoder, SentenceEncoder};
use thiserror::Error;
use udgraph::graph::Sentence;

use crate::encoders::{DependencyEncoder, EncoderType, EncodersConfig};

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    Immutable(ImmutableCategoricalEncoder<E, V>),
    Mutable(MutableCategoricalEncoder<E, V>),
}

impl<E, V> From<MutableCategoricalEncoder<E, V>> for CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    fn from(encoder: MutableCategoricalEncoder<E, V>) -> Self {
        CategoricalEncoderWrap::Mutable(encoder)
    }
}

impl<D> SentenceDecoder for CategoricalEncoderWrap<D, D::Encoding>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    type Error = D::Error;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        match self {
            CategoricalEncoderWrap::Immutable(decoder) => decoder.decode(labels, sentence),
            CategoricalEncoderWrap::Mutable(decoder) => decoder.decode(labels, sentence),
        }
    }
}

impl<E> SentenceEncoder for CategoricalEncoderWrap<E, E::Encoding>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    type Error = E::Error;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        match self {
            CategoricalEncoderWrap::Immutable(encoder) => encoder.encode(sentence),
            CategoricalEncoderWrap::Mutable(encoder) => encoder.encode(sentence),
        }
    }
}

impl<E, V> CategoricalEncoderWrap<E, V>
where
    V: Clone + Eq + Hash,
{
    pub fn len(&self) -> usize {
        match self {
            CategoricalEncoderWrap::Immutable(encoder) => encoder.len(),
            CategoricalEncoderWrap::Mutable(encoder) => encoder.len(),
        }
    }
}

/// Wrapper of encoder error types.
#[derive(Debug, Error)]
pub enum DecoderError {
    #[error(transparent)]
    Lemma(<EditTreeEncoder as SentenceDecoder>::Error),

    #[error(transparent)]
    Layer(<LayerEncoder as SentenceDecoder>::Error),

    #[error(transparent)]
    RelativePos(<RelativePosEncoder as SentenceDecoder>::Error),

    #[error(transparent)]
    RelativePosition(<RelativePositionEncoder as SentenceDecoder>::Error),

    #[error(transparent)]
    TdzLemma(<TdzLemmaEncoder as SentenceDecoder>::Error),
}

/// Wrapper of encoder error types.
#[derive(Debug, Error)]
pub enum EncoderError {
    #[error(transparent)]
    Lemma(<EditTreeEncoder as SentenceEncoder>::Error),

    #[error(transparent)]
    Layer(<LayerEncoder as SentenceEncoder>::Error),

    #[error(transparent)]
    RelativePos(<RelativePosEncoder as SentenceEncoder>::Error),

    #[error(transparent)]
    RelativePosition(<RelativePositionEncoder as SentenceEncoder>::Error),

    #[error(transparent)]
    TdzLemma(<TdzLemmaEncoder as SentenceEncoder>::Error),
}

/// Wrapper of the various supported encoders.
#[derive(Deserialize, Serialize)]
pub enum Encoder {
    Lemma(CategoricalEncoderWrap<EditTreeEncoder, EditTree>),
    Layer(CategoricalEncoderWrap<LayerEncoder, String>),
    RelativePos(CategoricalEncoderWrap<RelativePosEncoder, DependencyEncoding<RelativePos>>),
    RelativePosition(
        CategoricalEncoderWrap<RelativePositionEncoder, DependencyEncoding<RelativePosition>>,
    ),
    TdzLemma(CategoricalEncoderWrap<TdzLemmaEncoder, EditTree>),
}

#[allow(clippy::len_without_is_empty)]
impl Encoder {
    pub fn len(&self) -> usize {
        match self {
            Encoder::Layer(encoder) => encoder.len(),
            Encoder::Lemma(encoder) => encoder.len(),
            Encoder::RelativePos(encoder) => encoder.len(),
            Encoder::RelativePosition(encoder) => encoder.len(),
            Encoder::TdzLemma(encoder) => encoder.len(),
        }
    }
}

impl SentenceDecoder for Encoder {
    type Encoding = usize;

    type Error = DecoderError;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        match self {
            Encoder::Layer(decoder) => decoder
                .decode(labels, sentence)
                .map_err(DecoderError::Layer),
            Encoder::Lemma(decoder) => decoder
                .decode(labels, sentence)
                .map_err(DecoderError::Lemma),
            Encoder::RelativePos(decoder) => decoder
                .decode(labels, sentence)
                .map_err(DecoderError::RelativePos),
            Encoder::RelativePosition(decoder) => decoder
                .decode(labels, sentence)
                .map_err(DecoderError::RelativePosition),
            Encoder::TdzLemma(decoder) => decoder
                .decode(labels, sentence)
                .map_err(DecoderError::TdzLemma),
        }
    }
}

impl SentenceEncoder for Encoder {
    type Encoding = usize;

    type Error = EncoderError;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        match self {
            Encoder::Layer(encoder) => encoder.encode(sentence).map_err(EncoderError::Layer),
            Encoder::Lemma(encoder) => encoder.encode(sentence).map_err(EncoderError::Lemma),
            Encoder::RelativePos(encoder) => {
                encoder.encode(sentence).map_err(EncoderError::RelativePos)
            }
            Encoder::RelativePosition(encoder) => encoder
                .encode(sentence)
                .map_err(EncoderError::RelativePosition),
            Encoder::TdzLemma(encoder) => encoder.encode(sentence).map_err(EncoderError::TdzLemma),
        }
    }
}

impl From<&EncoderType> for Encoder {
    fn from(encoder_type: &EncoderType) -> Self {
        // We start labeling at 2. 0 is reserved for padding, 1 for continuations.
        match encoder_type {
            EncoderType::Dependency {
                encoder: DependencyEncoder::RelativePos(pos_layer),
                root_relation,
            } => Encoder::RelativePos(
                MutableCategoricalEncoder::new(
                    RelativePosEncoder::new(*pos_layer, root_relation),
                    Numberer::new(2),
                )
                .into(),
            ),
            EncoderType::Dependency {
                encoder: DependencyEncoder::RelativePosition,
                root_relation,
            } => Encoder::RelativePosition(
                MutableCategoricalEncoder::new(
                    RelativePositionEncoder::new(root_relation),
                    Numberer::new(2),
                )
                .into(),
            ),
            EncoderType::Lemma(backoff_strategy) => Encoder::Lemma(
                MutableCategoricalEncoder::new(
                    EditTreeEncoder::new(*backoff_strategy),
                    Numberer::new(2),
                )
                .into(),
            ),
            EncoderType::Sequence(ref layer) => Encoder::Layer(
                MutableCategoricalEncoder::new(LayerEncoder::new(layer.clone()), Numberer::new(2))
                    .into(),
            ),
            EncoderType::TdzLemma(backoff_strategy) => Encoder::TdzLemma(
                MutableCategoricalEncoder::new(
                    TdzLemmaEncoder::new(*backoff_strategy),
                    Numberer::new(2),
                )
                .into(),
            ),
        }
    }
}

/// A named encoder.
#[derive(Deserialize, Serialize)]
pub struct NamedEncoder {
    encoder: Encoder,
    name: String,
}

impl NamedEncoder {
    /// Get the encoder.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Get the encoder name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A collection of named encoders.
#[derive(Serialize, Deserialize)]
pub struct Encoders(Vec<NamedEncoder>);

impl From<&EncodersConfig> for Encoders {
    fn from(config: &EncodersConfig) -> Self {
        Encoders(
            config
                .iter()
                .map(|encoder| NamedEncoder {
                    name: encoder.name.clone(),
                    encoder: (&encoder.encoder).into(),
                })
                .collect(),
        )
    }
}

impl Deref for Encoders {
    type Target = [NamedEncoder];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
