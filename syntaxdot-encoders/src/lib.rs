//! Label encoders.

use std::error::Error;

use conllu::graph::Sentence;

pub mod categorical;

pub mod deprel;

pub mod layer;

pub mod lemma;

/// An encoding with its probability.
#[derive(Debug)]
pub struct EncodingProb<E> {
    encoding: E,
    prob: f32,
}

impl<E> EncodingProb<E>
where
    E: ToOwned,
{
    /// Create an encoding with its probability.
    ///
    /// This constructor takes an owned encoding.
    pub fn new(encoding: E, prob: f32) -> Self {
        EncodingProb { encoding, prob }
    }

    /// Get the encoding.
    pub fn encoding(&self) -> &E {
        &self.encoding
    }

    /// Get the probability of the encoding.
    pub fn prob(&self) -> f32 {
        self.prob
    }
}

impl<E> From<EncodingProb<E>> for (String, f32)
where
    E: Clone + ToString,
{
    fn from(prob: EncodingProb<E>) -> Self {
        (prob.encoding().to_string(), prob.prob())
    }
}

/// Trait for sentence decoders.
///
/// A sentence decoder adds a representation to each token in a
/// sentence, such as a part-of-speech tag or a topological field.
pub trait SentenceDecoder {
    type Encoding: ToOwned;

    /// The decoding error type.
    type Error: Error;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>;
}

/// Trait for sentence encoders.
///
/// A sentence encoder extracts a representation of each token in a
/// sentence, such as a part-of-speech tag or a topological field.
pub trait SentenceEncoder {
    type Encoding;

    /// The encoding error type.
    type Error: Error;

    /// Encode the given sentence.
    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error>;
}
