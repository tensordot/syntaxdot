use std::fmt;

use itertools::{multizip, Itertools};
use ndarray::{ArrayView2, Axis};
use numberer::Numberer;
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use udgraph::graph::{DepTriple, Node, Sentence};
use udgraph::token::Token;

use crate::categorical::{ImmutableNumberer, MutableNumberer, Number};
use crate::dependency::mst::chu_liu_edmonds;

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
        pairwise_head_scores: ArrayView2<f32>,
        best_pairwise_relations: ArrayView2<i32>,
        sentence: &mut Sentence,
    ) {
        let heads = chu_liu_edmonds(pairwise_head_scores.t(), 0);

        // Unwrap the heads, skipping the root vertex.
        let heads = heads
            .into_iter()
            .skip(1)
            .collect::<Option<Vec<usize>>>()
            // This should never happen.
            .expect("Non-root head without a parent?");

        let relations = heads
            .iter()
            .enumerate()
            .map(|(dep, &head)| best_pairwise_relations[(dep + 1, head)])
            .collect::<Vec<_>>();

        for (dep, head, relation) in multizip((1..sentence.len(), heads, relations)) {
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
                .add_deprel::<String>(DepTriple::new(head, Some(relation), dep));
        }
    }

    /// Greedily decode a dependency graph from a score matrix.
    ///
    /// The following arguments must be provided:
    ///
    /// * `pairwise_head_score`: edge (arc) score matrix, `pairwise_head_score[dependent][head]`
    ///   is the score for attaching `dependent` to `head`.
    /// * `best_pairwise_relations`: represents per dependent the best dependency relation
    ///   given a head (`best_pairwise_relations[dependent, head]`).
    /// * `sentence`: the sentence in which to store the dependency relations.
    pub fn decode_greedy(
        &self,
        pairwise_head_scores: ArrayView2<f32>,
        best_pairwise_relations: ArrayView2<i32>,
        sentence: &mut Sentence,
    ) {
        let heads = pairwise_head_scores
            .axis_iter(Axis(0))
            .skip(1)
            .map(|heads| {
                heads
                    .iter()
                    .map(|&v| NotNan::new(v).expect("Head score matrix contains NaN"))
                    .position_max()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let relations = heads
            .iter()
            .zip(best_pairwise_relations.axis_iter(Axis(0)).skip(1))
            .map(|(&head, best_relations)| best_relations[head])
            .collect::<Vec<_>>();

        for (dep, head, relation) in multizip((1..sentence.len(), heads, relations)) {
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
                .add_deprel(DepTriple::new(head, Some(relation), dep));
        }
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

    use conllu::io::Reader;
    use udgraph::graph::{DepTriple, Sentence};
    use udgraph::token::Token;

    use crate::dependency::{DependencyEncoding, EncodeError, MutableDependencyEncoder};
    use ndarray::Array2;

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
            .add_deprel(DepTriple::new(0, Some("root"), 2));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::<&str>::new(2, None, 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("obj"), 4));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(4, Some("det"), 3));

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
            .add_deprel(DepTriple::new(0, Some("root"), 2));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("nsubj"), 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("obj"), 4));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(4, Some("det"), 3));

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

            let head_scores = heads_to_scores(&encoding.heads);
            let best_relations = relations_to_matrix(&encoding.heads, &encoding.relations);

            let mut decoded_sentence = sentence.clone();

            // Test MST decoding.
            encoder.decode(
                head_scores.view(),
                best_relations.view(),
                &mut decoded_sentence,
            );
            assert_eq!(decoded_sentence, sentence);

            // Test greedy decoding.
            encoder.decode_greedy(
                head_scores.view(),
                best_relations.view(),
                &mut decoded_sentence,
            );
            assert_eq!(decoded_sentence, sentence);
        }
    }

    fn heads_to_scores(heads: &[usize]) -> Array2<f32> {
        // Number of tokens, including root.
        let n_tokens = heads.len() + 1;

        Array2::from_shape_fn((n_tokens, n_tokens), |(dep, head)| {
            if dep == 0 {
                0.0
            } else if heads[dep - 1] == head {
                1.0
            } else {
                0.0
            }
        })
    }

    fn relations_to_matrix(heads: &[usize], relations: &[usize]) -> Array2<i32> {
        // Number of tokens, including root.
        let n_tokens = heads.len() + 1;

        Array2::from_shape_fn((n_tokens, n_tokens), |(dep, head)| {
            if dep == 0 {
                -1
            } else if heads[dep - 1] == head {
                relations[dep - 1] as i32
            } else {
                -1
            }
        })
    }
}
