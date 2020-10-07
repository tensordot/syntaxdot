use std::convert::Infallible;

use conllu::graph::{DepTriple, Sentence};
use serde_derive::{Deserialize, Serialize};

use super::{
    attach_orphans, break_cycles, find_or_create_root, DecodeError, DependencyEncoding, EncodeError,
};
use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

/// Relative head position.
///
/// The position of the head relative to the dependent token.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RelativePosition(isize);

impl ToString for DependencyEncoding<RelativePosition> {
    fn to_string(&self) -> String {
        format!("{}/{}", self.label, self.head.0)
    }
}

/// Relative position encoder.
///
/// This encoder encodes dependency relations as token labels. The
/// dependency relation is encoded as-is. The position of the head
/// is encoded relative to the (dependent) token.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RelativePositionEncoder {
    root_relation: String,
}

impl RelativePositionEncoder {
    pub fn new(root_relation: impl Into<String>) -> Self {
        RelativePositionEncoder {
            root_relation: root_relation.into(),
        }
    }
}

impl RelativePositionEncoder {
    fn decode_idx(
        idx: usize,
        sentence_len: usize,
        encoding: &DependencyEncoding<RelativePosition>,
    ) -> Result<DepTriple<String>, DecodeError> {
        let DependencyEncoding {
            label,
            head: RelativePosition(head),
        } = encoding;

        let head_idx = idx as isize + head;
        if head_idx < 0 || head_idx >= sentence_len as isize {
            return Err(DecodeError::PositionOutOfBounds);
        }

        Ok(DepTriple::new(
            (idx as isize + head) as usize,
            Some(label.clone()),
            idx,
        ))
    }
}

impl SentenceEncoder for RelativePositionEncoder {
    type Encoding = DependencyEncoding<RelativePosition>;

    type Error = EncodeError;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        let mut encoded = Vec::with_capacity(sentence.len());
        for idx in 1..sentence.len() {
            let triple = sentence
                .dep_graph()
                .head(idx)
                .ok_or_else(|| EncodeError::missing_head(idx, sentence))?;
            let relation = triple
                .relation()
                .ok_or_else(|| EncodeError::missing_relation(idx, sentence))?;

            encoded.push(DependencyEncoding {
                label: relation.to_owned(),
                head: RelativePosition(triple.head() as isize - triple.dependent() as isize),
            });
        }

        Ok(encoded)
    }
}

impl SentenceDecoder for RelativePositionEncoder {
    type Encoding = DependencyEncoding<RelativePosition>;

    type Error = Infallible;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        let token_indices: Vec<_> = (0..sentence.len())
            .filter(|&idx| sentence[idx].is_token())
            .collect();

        for (idx, encodings) in token_indices.into_iter().zip(labels) {
            for encoding in encodings.as_ref() {
                if let Ok(triple) =
                    RelativePositionEncoder::decode_idx(idx, sentence.len(), encoding.encoding())
                {
                    sentence.dep_graph_mut().add_deprel(triple);
                    break;
                }
            }
        }

        // Fixup tree.
        let sentence_len = sentence.len();
        let root_idx = find_or_create_root(
            labels,
            sentence,
            |idx, encoding| Self::decode_idx(idx, sentence_len, encoding).ok(),
            &self.root_relation,
        );
        attach_orphans(labels, sentence, root_idx);
        break_cycles(sentence, root_idx);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use conllu::graph::{DepTriple, Sentence};
    use conllu::token::TokenBuilder;

    use super::{RelativePosition, RelativePositionEncoder};
    use crate::deprel::{DecodeError, DependencyEncoding};
    use crate::{EncodingProb, SentenceDecoder};

    const ROOT_RELATION: &str = "root";

    // Small tests for the relative position encoder. Automatic
    // testing is performed in the module tests.

    #[test]
    fn position_out_of_bounds() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("a").xpos("A").into());
        sent.push(TokenBuilder::new("b").xpos("B").into());

        assert_eq!(
            RelativePositionEncoder::decode_idx(
                1,
                sent.len(),
                &DependencyEncoding {
                    label: "X".into(),
                    head: RelativePosition(-2),
                },
            ),
            Err(DecodeError::PositionOutOfBounds)
        )
    }

    #[test]
    fn backoff() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("a").xpos("A").into());

        let decoder = RelativePositionEncoder::new(ROOT_RELATION);
        let labels = vec![vec![
            EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.into(),
                    head: RelativePosition(-2),
                },
                1.0,
            ),
            EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.into(),
                    head: RelativePosition(-1),
                },
                1.0,
            ),
        ]];

        decoder.decode(&labels, &mut sent).unwrap();

        assert_eq!(
            sent.dep_graph().head(1),
            Some(DepTriple::new(0, Some(ROOT_RELATION), 1))
        );
    }
}
