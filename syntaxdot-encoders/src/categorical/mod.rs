use std::hash::Hash;
use std::marker::PhantomData;

use conllu::graph::Sentence;
use numberer::Numberer;
use serde_derive::{Deserialize, Serialize};

use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

mod mutability {
    use std::cell::RefCell;
    use std::hash::Hash;

    use numberer::Numberer;
    use serde_derive::{Deserialize, Serialize};

    pub trait Number<V>
    where
        V: Clone + Eq + Hash,
    {
        fn new(numberer: Numberer<V>) -> Self;

        fn len(&self) -> usize;

        fn number(&self, value: V) -> Option<usize>;

        fn value(&self, number: usize) -> Option<V>;
    }

    #[derive(Deserialize, Serialize)]
    pub struct ImmutableNumberer<V>(Numberer<V>)
    where
        V: Clone + Eq + Hash;

    impl<V> Number<V> for ImmutableNumberer<V>
    where
        V: Clone + Eq + Hash,
    {
        fn new(numberer: Numberer<V>) -> Self {
            ImmutableNumberer(numberer)
        }

        fn len(&self) -> usize {
            self.0.len()
        }

        fn number(&self, value: V) -> Option<usize> {
            self.0.number(&value)
        }

        fn value(&self, number: usize) -> Option<V> {
            self.0.value(number).cloned()
        }
    }

    #[derive(Deserialize, Serialize)]
    pub struct MutableNumberer<V>(RefCell<Numberer<V>>)
    where
        V: Clone + Eq + Hash;

    impl<V> Number<V> for MutableNumberer<V>
    where
        V: Clone + Eq + Hash,
    {
        fn new(numberer: Numberer<V>) -> Self {
            MutableNumberer(RefCell::new(numberer))
        }

        fn len(&self) -> usize {
            self.0.borrow().len()
        }

        fn number(&self, value: V) -> Option<usize> {
            Some(self.0.borrow_mut().add(value))
        }

        fn value(&self, number: usize) -> Option<V> {
            self.0.borrow().value(number).cloned()
        }
    }
}

/// An immutable categorical encoder
///
/// This encoder does not add new encodings to the encoder. If the
/// number of an unknown encoding is looked up, the special value `0`
/// is used.
pub type ImmutableCategoricalEncoder<E, V> =
    CategoricalEncoder<E, V, mutability::ImmutableNumberer<V>>;

/// A mutable categorical encoder
///
/// This encoder adds new encodings to the encoder when encountered
pub type MutableCategoricalEncoder<E, V> = CategoricalEncoder<E, V, mutability::MutableNumberer<V>>;

/// An encoder wrapper that encodes/decodes to a categorical label.
#[derive(Deserialize, Serialize)]
pub struct CategoricalEncoder<E, V, M>
where
    V: Clone + Eq + Hash,
    M: mutability::Number<V>,
{
    inner: E,
    numberer: M,

    #[serde(skip)]
    _phantom: PhantomData<V>,
}

impl<E, V, M> CategoricalEncoder<E, V, M>
where
    V: Clone + Eq + Hash,
    M: mutability::Number<V>,
{
    pub fn new(encoder: E, numberer: Numberer<V>) -> Self {
        CategoricalEncoder {
            inner: encoder,
            numberer: M::new(numberer),
            _phantom: PhantomData,
        }
    }
}

impl<D, M> CategoricalEncoder<D, D::Encoding, M>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash + ToOwned,
    M: mutability::Number<D::Encoding>,
{
    /// Decode without applying the inner decoder.
    pub fn decode_without_inner<S>(&self, labels: &[S]) -> Vec<Vec<EncodingProb<D::Encoding>>>
    where
        S: AsRef<[EncodingProb<usize>]>,
    {
        labels
            .iter()
            .map(|encoding_probs| {
                encoding_probs
                    .as_ref()
                    .iter()
                    .map(|encoding_prob| {
                        EncodingProb::new(
                            self.numberer
                                .value(*encoding_prob.encoding())
                                .expect("Unknown label"),
                            encoding_prob.prob(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }
}

impl<E, V, M> CategoricalEncoder<E, V, M>
where
    V: Clone + Eq + Hash,
    M: mutability::Number<V>,
{
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.numberer.len()
    }
}

impl<E, M> SentenceEncoder for CategoricalEncoder<E, E::Encoding, M>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    M: mutability::Number<E::Encoding>,
{
    type Encoding = usize;

    type Error = E::Error;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        let encoding = self.inner.encode(sentence)?;
        let categorical_encoding = encoding
            .into_iter()
            .map(|e| self.numberer.number(e).unwrap_or(0))
            .collect();
        Ok(categorical_encoding)
    }
}

impl<D, M> SentenceDecoder for CategoricalEncoder<D, D::Encoding, M>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
    M: mutability::Number<D::Encoding>,
{
    type Encoding = usize;

    type Error = D::Error;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        let categorial_encoding = self.decode_without_inner(labels);
        self.inner.decode(&categorial_encoding, sentence)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use conllu::io::Reader;
    use numberer::Numberer;

    use super::{EncodingProb, MutableCategoricalEncoder, SentenceDecoder, SentenceEncoder};
    use crate::layer::Layer;
    use crate::layer::LayerEncoder;

    static NON_PROJECTIVE_DATA: &'static str = "testdata/lassy-small-dev.conllu";

    fn test_encoding<P, E, C>(path: P, encoder_decoder: E)
    where
        P: AsRef<Path>,
        E: SentenceEncoder<Encoding = C> + SentenceDecoder<Encoding = C>,
        C: 'static + Clone,
    {
        let f = File::open(path).unwrap();
        let reader = Reader::new(BufReader::new(f));

        for sentence in reader {
            let sentence = sentence.unwrap();

            // Encode
            let encodings = encoder_decoder
                .encode(&sentence)
                .unwrap()
                .into_iter()
                .map(|e| [EncodingProb::new(e, 1.)])
                .collect::<Vec<_>>();

            // Decode
            let mut test_sentence = sentence.clone();
            encoder_decoder
                .decode(&encodings, &mut test_sentence)
                .unwrap();

            assert_eq!(sentence, test_sentence);
        }
    }

    #[test]
    fn categorical_encoder() {
        let numberer = Numberer::new(1);
        let encoder = LayerEncoder::new(Layer::XPos);
        let categorical_encoder = MutableCategoricalEncoder::new(encoder, numberer);
        assert_eq!(categorical_encoder.len(), 1);
        test_encoding(NON_PROJECTIVE_DATA, categorical_encoder);
    }
}
