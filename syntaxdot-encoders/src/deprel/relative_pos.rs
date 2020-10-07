// Implementation note:
//
// We currently do a binary search to find the position of the token
// in the POS table. This makes encoding/decoding of a sentence
// *O(n log(n))*. We could do this in *O(n)* by keeping track of
// the position of the token the head POS table while constructing
// the POS table. This currently does not really seem worth it?

use std::convert::Infallible;

use std::collections::HashMap;

use conllu::graph::{DepTriple, Node, Sentence};
use conllu::token::Token;
use serde_derive::{Deserialize, Serialize};

use super::{
    attach_orphans, break_cycles, find_or_create_root, DecodeError, DependencyEncoding, EncodeError,
};
use crate::{EncodingProb, SentenceDecoder, SentenceEncoder};

const ROOT_POS: &str = "ROOT";

/// Part-of-speech layer.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum POSLayer {
    /// Universal part-of-speech tag.
    UPos,

    /// Language-specific part-of-speech tag.
    XPos,
}

impl POSLayer {
    fn pos(self, token: &Token) -> Option<&str> {
        match self {
            POSLayer::UPos => token.upos(),
            POSLayer::XPos => token.xpos(),
        }
    }
}

/// Relative head position by part-of-speech.
///
/// The position of the head relative to the dependent token,
/// in terms of part-of-speech tags. For example, a position of
/// *-2* with the pos *noun* means that the head is the second
/// preceding noun.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RelativePOS {
    pos: String,
    position: isize,
}

impl RelativePOS {
    #[allow(dead_code)]
    pub fn new(pos: impl Into<String>, position: isize) -> Self {
        RelativePOS {
            pos: pos.into(),
            position,
        }
    }
}

impl ToString for DependencyEncoding<RelativePOS> {
    fn to_string(&self) -> String {
        format!("{}/{}/{}", self.label, self.head.pos, self.head.position)
    }
}

/// Relative part-of-speech position encoder.
///
/// This encoder encodes dependency relations as token labels. The
/// dependency relation is encoded as-is. The position of the head
/// is encoded relative to the (dependent) token by part-of-speech.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct RelativePOSEncoder {
    pos_layer: POSLayer,
    root_relation: String,
}

impl RelativePOSEncoder {
    pub fn new(pos_layer: POSLayer, root_relation: impl Into<String>) -> Self {
        RelativePOSEncoder {
            pos_layer,
            root_relation: root_relation.into(),
        }
    }
}

impl RelativePOSEncoder {
    pub(crate) fn decode_idx(
        pos_table: &HashMap<String, Vec<usize>>,
        idx: usize,
        encoding: &DependencyEncoding<RelativePOS>,
    ) -> Result<DepTriple<String>, DecodeError> {
        let DependencyEncoding { label, head } = encoding;

        let indices = pos_table
            .get(head.pos.as_str())
            .ok_or(DecodeError::InvalidPOS)?;

        let head_idx = Self::head_index(indices, idx, head.position)?;

        Ok(DepTriple::new(head_idx, Some(label.to_owned()), idx))
    }

    /// Find the relative position of a dependent to a head.
    ///
    /// This methods finds the relative position of `dependent` to
    /// `head` in `indices`.
    fn relative_dependent_position(indices: &[usize], head: usize, dependent: usize) -> isize {
        let mut head_position = indices
            .binary_search(&head)
            .expect("Head is missing in sorted POS tag list");

        let dependent_position = match indices.binary_search(&dependent) {
            Ok(idx) => idx,
            Err(idx) => {
                // The head moves one place if the dependent is inserted
                // before the head. Consider e.g. the indices
                //
                // [3, 6, 9]
                //     ^--- insertion point of 4.
                //
                // Suppose that we want to compute the relative
                // position of 4 to its head 9 (position 2). The
                // insertion point is 1. When computing the relative
                // position, we should take into account that 4 lies
                // before 6.
                if dependent < head {
                    head_position += 1;
                }
                idx
            }
        };

        head_position as isize - dependent_position as isize
    }

    /// Get the index of the head of `dependent`.
    ///
    /// Get index of the head of `dependent`, given the relative
    /// position of `dependent` to the head in `indices`.
    fn head_index(
        indices: &[usize],
        dependent: usize,
        mut relative_head_position: isize,
    ) -> Result<usize, DecodeError> {
        let dependent_position = match indices.binary_search(&dependent) {
            Ok(idx) => idx,
            Err(idx) => {
                // Consider e.g. the indices
                //
                // [3, 6, 9]
                //     ^--- insertion point of 4.
                //
                // Suppose that 4 is the dependent and +2 the relative
                // position of the head. The relative position takes
                // both succeeding elements (6, 9) into
                // account. However, the insertion point is the
                // element at +1. So, compensate for this in the
                // relative position.
                if relative_head_position > 0 {
                    relative_head_position -= 1
                }
                idx
            }
        };

        let head_position = dependent_position as isize + relative_head_position;
        if head_position < 0 || head_position >= indices.len() as isize {
            return Err(DecodeError::PositionOutOfBounds);
        }

        Ok(indices[head_position as usize])
    }

    pub(crate) fn pos_position_table(&self, sentence: &Sentence) -> HashMap<String, Vec<usize>> {
        let mut table: HashMap<String, Vec<_>> = HashMap::new();

        for (idx, node) in sentence.iter().enumerate() {
            let pos = match node {
                Node::Root => ROOT_POS.into(),
                Node::Token(token) => match self.pos_layer.pos(token) {
                    Some(pos) => pos.into(),
                    None => continue,
                },
            };

            let indices = table.entry(pos).or_default();
            indices.push(idx);
        }

        table
    }
}

impl SentenceEncoder for RelativePOSEncoder {
    type Encoding = DependencyEncoding<RelativePOS>;

    type Error = EncodeError;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Self::Error> {
        let pos_table = self.pos_position_table(&sentence);

        let mut encoded = Vec::with_capacity(sentence.len());
        for idx in 0..sentence.len() {
            if let Node::Root = &sentence[idx] {
                continue;
            }

            let triple = sentence
                .dep_graph()
                .head(idx)
                .ok_or_else(|| EncodeError::missing_head(idx, sentence))?;
            let relation = triple
                .relation()
                .ok_or_else(|| EncodeError::missing_relation(idx, sentence))?;

            let head_pos = match &sentence[triple.head()] {
                Node::Root => ROOT_POS,
                Node::Token(head_token) => self
                    .pos_layer
                    .pos(head_token)
                    .ok_or_else(|| EncodeError::missing_pos(idx, sentence))?,
            };

            let position = Self::relative_dependent_position(
                &pos_table[head_pos],
                triple.head(),
                triple.dependent(),
            );

            encoded.push(DependencyEncoding {
                label: relation.to_owned(),
                head: RelativePOS {
                    pos: head_pos.to_owned(),
                    position,
                },
            });
        }

        Ok(encoded)
    }
}

impl SentenceDecoder for RelativePOSEncoder {
    type Encoding = DependencyEncoding<RelativePOS>;

    type Error = Infallible;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Self::Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
    {
        let pos_table = self.pos_position_table(&sentence);

        let token_indices: Vec<_> = (0..sentence.len())
            .filter(|&idx| sentence[idx].is_token())
            .collect();

        for (idx, encodings) in token_indices.into_iter().zip(labels) {
            for encoding in encodings.as_ref() {
                if let Ok(triple) =
                    RelativePOSEncoder::decode_idx(&pos_table, idx, encoding.encoding())
                {
                    sentence.dep_graph_mut().add_deprel(triple);
                    break;
                }
            }
        }

        // Fixup tree.
        let root_idx = find_or_create_root(
            labels,
            sentence,
            |idx, encoding| Self::decode_idx(&pos_table, idx, encoding).ok(),
            &self.root_relation,
        );
        attach_orphans(labels, sentence, root_idx);
        break_cycles(sentence, root_idx);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::iter::FromIterator;

    use conllu::graph::{DepTriple, Sentence};
    use conllu::token::TokenBuilder;

    use super::{POSLayer, RelativePOS, RelativePOSEncoder, ROOT_POS};
    use crate::deprel::{DecodeError, DependencyEncoding};
    use crate::{EncodingProb, SentenceDecoder};

    const ROOT_RELATION: &str = "root";

    // Small tests for relative part-of-speech encoder. Automatic
    // testing is performed in the module tests.

    #[test]
    fn invalid_pos() {
        assert_eq!(
            RelativePOSEncoder::decode_idx(
                &HashMap::new(),
                0,
                &DependencyEncoding {
                    label: "X".into(),
                    head: RelativePOS {
                        pos: "C".into(),
                        position: -1,
                    },
                },
            ),
            Err(DecodeError::InvalidPOS)
        )
    }

    #[test]
    fn position_out_of_bounds() {
        assert_eq!(
            RelativePOSEncoder::decode_idx(
                &HashMap::from_iter(vec![("A".to_string(), vec![0])]),
                1,
                &DependencyEncoding {
                    label: "X".into(),
                    head: RelativePOS {
                        pos: "A".into(),
                        position: -2,
                    },
                },
            ),
            Err(DecodeError::PositionOutOfBounds)
        )
    }

    #[test]
    fn backoff() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("a").xpos("A").into());

        let decoder = RelativePOSEncoder::new(POSLayer::XPos, ROOT_RELATION);
        let labels = vec![vec![
            EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.into(),
                    head: RelativePOS {
                        pos: ROOT_POS.into(),
                        position: -2,
                    },
                },
                1.0,
            ),
            EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.into(),
                    head: RelativePOS {
                        pos: ROOT_POS.into(),
                        position: -1,
                    },
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
