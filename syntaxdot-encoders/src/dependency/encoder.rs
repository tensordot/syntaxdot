use std::fmt;

use itertools::multizip;
use ndarray::{s, ArrayView1};
use numberer::Numberer;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use udgraph::graph::{DepTriple, Node, Sentence};
use udgraph::token::Token;
use udgraph::Error;

use crate::categorical::{ImmutableNumberer, MutableNumberer, Number};

/// Dependency encoding.
#[derive(Debug, Eq, PartialEq)]
pub struct DependencyEncoding {
    /// The head of each (non-ROOT) token.
    pub heads: Vec<usize>,

    /// The dependency relation of each (non-ROOT) token.
    pub relations: Vec<usize>,
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EncodeError {
    /// The token does not have a head.
    MissingHead { token: usize, sent: Vec<String> },

    /// The token does not have a dependency relation.
    MissingRelation { token: usize, sent: Vec<String> },
}

impl EncodeError {
    /// Construct `EncodeError::MissingHead` from a CoNLL-U graph.
    ///
    /// Construct an error. `token` is the node index for which the
    /// error applies in `sentence`.
    pub fn missing_head(token: usize, sentence: &Sentence) -> Self {
        Self::MissingHead {
            sent: Self::sentence_to_forms(sentence),
            token: token - 1,
        }
    }

    /// Construct `EncodeError::MissingRelation` from a CoNLL-X graph.
    ///
    /// Construct an error. `token` is the node index for which the
    /// error applies in `sentence`.
    pub fn missing_relation(token: usize, sentence: &Sentence) -> Self {
        Self::MissingRelation {
            sent: Self::sentence_to_forms(sentence),
            token: token - 1,
        }
    }

    fn format_bracketed(bracket_idx: usize, tokens: &[String]) -> String {
        let mut tokens = tokens.to_owned();
        tokens.insert(bracket_idx + 1, "]".to_string());
        tokens.insert(bracket_idx, "[".to_string());

        tokens.join(" ")
    }

    fn sentence_to_forms(sentence: &Sentence) -> Vec<String> {
        sentence
            .iter()
            .filter_map(Node::token)
            .map(Token::form)
            .map(ToOwned::to_owned)
            .collect()
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use EncodeError::*;

        match self {
            MissingHead { token, sent } => write!(
                f,
                "Token does not have a head:\n\n{}\n",
                Self::format_bracketed(*token, sent),
            ),
            MissingRelation { token, sent } => write!(
                f,
                "Token does not have a dependency relation:\n\n{}\n",
                Self::format_bracketed(*token, sent),
            ),
        }
    }
}

/// Arc-factored dependency encoder/decoder.
#[derive(Serialize, Deserialize)]
pub struct DependencyEncoder<N>
where
    N: Number<String>,
{
    relations: N,
}

impl<N> DependencyEncoder<N>
where
    N: Number<String>,
{
    /// Encode a sentence.
    ///
    /// Returns the encoding of the dependency graph.
    pub fn encode(&self, sentence: &Sentence) -> Result<DependencyEncoding, EncodeError> {
        let dep_graph = sentence.dep_graph();

        let mut heads = Vec::with_capacity(sentence.len());
        let mut relations = Vec::with_capacity(sentence.len());

        for token_idx in 1..sentence.len() {
            let head = dep_graph
                .head(token_idx)
                .ok_or_else(|| EncodeError::missing_head(token_idx, sentence))?
                .head();
            heads.push(head);

            let relation = dep_graph
                .head(token_idx)
                .and_then(|triple| triple.relation().map(ToString::to_string))
                .ok_or_else(|| EncodeError::missing_relation(token_idx, sentence))?;
            relations.push(
                self.relations
                    .number(relation.to_string())
                    .expect("Unknown dependency relation"),
            );
        }

        Ok(DependencyEncoding { heads, relations })
    }

    /// Decode a dependency graph from a score matrix.
    ///
    /// The following arguments must be provided:
    ///
    /// * `pairwise_head_score`: edge (arc) score matrix, `pairwise_head_score[dependent][head]`
    ///   is the score for attaching `dependent` to `head`.
    /// * `best_pairwise_relations`: represents per dependent the best dependency relation
    ///   given a head (`best_pairwise_relations[dependent, head]`).
    /// * `sentence`: the sentence in which to store the dependency relations.
    pub fn decode(
        &self,
        sent_heads: ArrayView1<i64>,
        best_pairwise_relations: ArrayView1<i32>,
        sentence: &mut Sentence,
    ) -> Result<(), Error> {
        // Unwrap the heads, skipping the root vertex.
        let heads = sent_heads.slice(s![1..]);

        let relations = best_pairwise_relations
            .into_iter()
            .skip(1)
            .cloned()
            .collect::<Vec<_>>();

        for (dep, &head, relation) in multizip((1..sentence.len(), heads, relations)) {
            let relation = self
                .relations
                .value(relation as usize)
                // We should never predict an unknown relation, that would mean that
                // the model does not correspond to the label inventory. This cannot
                // happen, because the model's shape is based on the number of relations
                // reported by instances of this type.
                .unwrap_or_else(|| panic!("Predicted an unknown relation: {}", relation));
            sentence
                .dep_graph_mut()
                .add_deprel::<String>(DepTriple::new(head as usize, Some(relation), dep))?;
        }

        Ok(())
    }

    pub fn n_relations(&self) -> usize {
        self.relations.len()
    }
}

pub type ImmutableDependencyEncoder = DependencyEncoder<ImmutableNumberer<String>>;

pub type MutableDependencyEncoder = DependencyEncoder<MutableNumberer<String>>;

impl Default for MutableDependencyEncoder {
    fn default() -> Self {
        DependencyEncoder {
            relations: MutableNumberer::new(Numberer::new(0)),
        }
    }
}

impl MutableDependencyEncoder {
    /// Create a mutable dependency encoder.
    pub fn new() -> Self {
        Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::iter::once;

    use conllu::io::Reader;
    use udgraph::graph::{DepTriple, Sentence};
    use udgraph::token::Token;

    use crate::dependency::{DependencyEncoding, EncodeError, MutableDependencyEncoder};
    use ndarray::Array1;

    static NON_PROJECTIVE_DATA: &str = "testdata/lassy-small-dev.conllu";

    #[test]
    pub fn encoding_fails_with_missing_head() {
        let sent: Sentence = vec![
            Token::new("Ze"),
            Token::new("koopt"),
            Token::new("een"),
            Token::new("auto"),
        ]
        .into_iter()
        .collect();

        let encoder = MutableDependencyEncoder::new();

        assert!(matches!(
            encoder.encode(&sent),
            Err(EncodeError::MissingHead { .. })
        ));
    }

    #[test]
    pub fn encoding_fails_with_missing_relation() {
        let mut sent: Sentence = vec![
            Token::new("Ze"),
            Token::new("koopt"),
            Token::new("een"),
            Token::new("auto"),
        ]
        .into_iter()
        .collect();

        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(0, Some("root"), 2))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::<&str>::new(2, None, 1))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("obj"), 4))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(4, Some("det"), 3))
            .unwrap();

        let encoder = MutableDependencyEncoder::new();

        assert!(matches!(
            encoder.encode(&sent),
            Err(EncodeError::MissingRelation { .. })
        ));
    }

    #[test]
    pub fn encoder_encodes_correctly() {
        let mut sent: Sentence = vec![
            Token::new("Ze"),
            Token::new("koopt"),
            Token::new("een"),
            Token::new("auto"),
        ]
        .into_iter()
        .collect();

        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(0, Some("root"), 2))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("nsubj"), 1))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("obj"), 4))
            .unwrap();
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(4, Some("det"), 3))
            .unwrap();

        let encoder = MutableDependencyEncoder::new();

        let encoding = encoder.encode(&sent).unwrap();

        assert_eq!(
            encoding,
            DependencyEncoding {
                heads: vec![2, 0, 4, 2],
                relations: vec![0, 1, 2, 3]
            }
        )
    }

    #[test]
    pub fn no_changes_in_encode_decode_roundtrip() {
        let f = File::open(NON_PROJECTIVE_DATA).unwrap();
        let reader = Reader::new(BufReader::new(f));

        let encoder = MutableDependencyEncoder::new();

        for sentence in reader {
            let sentence = sentence.unwrap();
            let encoding = encoder.encode(&sentence).unwrap();

            let heads = once(0)
                .chain(encoding.heads.into_iter().map(|v| v as i64))
                .collect::<Array1<_>>();
            let best_relations = once(-1)
                .chain(encoding.relations.into_iter().map(|v| v as i32))
                .collect::<Array1<_>>();

            let mut decoded_sentence = sentence.clone();

            // Test MST decoding.
            encoder
                .decode(heads.view(), best_relations.view(), &mut decoded_sentence)
                .unwrap();
            assert_eq!(decoded_sentence, sentence);
        }
    }
}
