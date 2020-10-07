use conllu::graph::Sentence;
use lazy_static::lazy_static;
use ohnomore::transform::delemmatization::{
    RemoveAlternatives, RemoveReflexiveTag, RemoveSepVerbPrefix, RemoveTruncMarker,
};
use ohnomore::transform::lemmatization::{
    AddReflexiveTag, AddSeparatedVerbPrefix, FormAsLemma, MarkVerbPrefix, RestoreCase,
};
use ohnomore::transform::misc::{
    SimplifyArticleLemma, SimplifyPIAT, SimplifyPIDAT, SimplifyPIS, SimplifyPossesivePronounLemma,
};
use ohnomore::transform::Transforms;
use serde::{Deserialize, Serialize};

use crate::lemma::{BackoffStrategy, EditTreeEncoder};
use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

lazy_static! {
    static ref DECODE_TRANSFORMS: Transforms = {
        Transforms(vec![
            Box::new(FormAsLemma),
            Box::new(RestoreCase),
            Box::new(AddReflexiveTag),
            Box::new(AddSeparatedVerbPrefix::new(true)),
            Box::new(MarkVerbPrefix::new()),
            Box::new(SimplifyArticleLemma),
            Box::new(SimplifyPossesivePronounLemma),
            Box::new(SimplifyPIS),
            Box::new(SimplifyPIDAT),
            Box::new(SimplifyPIAT),
        ])
    };
    static ref ENCODE_TRANSFORMS: Transforms = {
        Transforms(vec![
            Box::new(RemoveAlternatives),
            Box::new(RemoveReflexiveTag),
            Box::new(RemoveSepVerbPrefix),
            Box::new(RemoveTruncMarker),
            Box::new(SimplifyArticleLemma),
            Box::new(SimplifyPossesivePronounLemma),
            Box::new(FormAsLemma),
        ])
    };
}

/// Lemma encoder-decoder for TüBa-D/Z
///
/// This encoder wraps `EditTreeEncoder`. Before encoding and after
/// decoding a list of transformation rules is applied to transform
/// the lemmas from and to TüBa-D/Z-style lemmas.
///
/// For example, the particle verb *abschließen* is encoded as the
/// *ab#schließen* in TüBa-D/Z. During encoding, the lemma is
/// transformed to *schließen*. Then during decoding the lemma is
/// transformed back to *ab#schließen* based on the *ab* particle that
/// is either a prefix of in form (e.g. *abgeschlossen*) or a
/// separated particle (e.g. *ich schließe es ab*).
#[derive(Deserialize, Serialize)]
#[serde(transparent)]
pub struct TdzLemmaEncoder {
    inner: EditTreeEncoder,
}

impl TdzLemmaEncoder {
    /// Construct a `TdzLemmaEncoder`.
    ///
    /// The backoff strategy is used when the edit tree that was
    /// predicted is not applicable to the form.
    pub fn new(backoff_strategy: BackoffStrategy) -> Self {
        TdzLemmaEncoder {
            inner: EditTreeEncoder::new(backoff_strategy),
        }
    }
}

impl SentenceDecoder for TdzLemmaEncoder {
    type Encoding = <EditTreeEncoder as SentenceDecoder>::Encoding;

    type Error = <EditTreeEncoder as SentenceDecoder>::Error;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        // Decode edit trees.
        self.inner.decode(labels, sentence)?;

        // Apply TüBa-D/Z transformations
        DECODE_TRANSFORMS.transform(sentence);

        Ok(())
    }
}

impl SentenceEncoder for TdzLemmaEncoder {
    type Encoding = <EditTreeEncoder as SentenceEncoder>::Encoding;

    type Error = <EditTreeEncoder as SentenceEncoder>::Error;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        // Hmpf, but we need to modify the sentence in-place.
        let mut sentence = sentence.clone();

        // Apply tranformations to remove TüBa-D/Z specifics.
        ENCODE_TRANSFORMS.transform(&mut sentence);

        self.inner.encode(&sentence)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use conllu::graph::{DepTriple, Sentence};
    use conllu::token::TokenBuilder;
    use edit_tree::EditTree as EditTreeInner;

    use super::TdzLemmaEncoder;
    use crate::lemma::BackoffStrategy;
    use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

    fn example_sentence() -> Sentence {
        let tokens = vec![
            TokenBuilder::new("Ich")
                .upos("PRON")
                .xpos("PPER")
                .lemma("ich")
                .into(),
            TokenBuilder::new("reise")
                .upos("VERB")
                .xpos("VVFIN")
                .lemma("ab#reisen")
                .into(),
            TokenBuilder::new("ab")
                .upos("ADP")
                .xpos("PTKVZ")
                .lemma("ab")
                .into(),
        ];

        let mut sent = Sentence::from_iter(tokens);

        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("compound:prt"), 3));

        sent
    }

    fn sentence_edit_trees() -> Vec<EditTreeInner<char>> {
        vec![
            EditTreeInner::create_tree(&['I', 'c', 'h'], &['i', 'c', 'h']),
            EditTreeInner::create_tree(&['r', 'e', 'i', 's', 'e'], &['r', 'e', 'i', 's', 'e', 'n']),
            EditTreeInner::create_tree(&['a', 'b'], &['a', 'b']),
        ]
    }

    fn encode_and_wrap(
        encoder: &TdzLemmaEncoder,
        sent: &Sentence,
    ) -> Vec<Vec<EncodingProb<EditTreeInner<char>>>> {
        encoder
            .encode(&sent)
            .unwrap()
            .into_iter()
            .map(|encoding| vec![EncodingProb::new(encoding, 1.0)])
            .collect::<Vec<_>>()
    }

    #[test]
    fn encodes_with_transformations() {
        let sent = example_sentence();

        let encoder = TdzLemmaEncoder::new(BackoffStrategy::Nothing);

        // Check whether the encoder transformations are applied.
        let encoding = encoder.encode(&sent).unwrap();
        assert_eq!(encoding, sentence_edit_trees());

        let encoding = encode_and_wrap(&encoder, &sent);

        let mut sent_decoded = sent.clone();
        encoder.decode(&encoding, &mut sent_decoded).unwrap();

        // Check whether the encoder transformations are applied.
        assert_eq!(sent, sent_decoded);
    }
}
