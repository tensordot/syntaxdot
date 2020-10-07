use std::fmt;

use conllu::graph::{Node, Sentence};
use conllu::token::Token;
use thiserror::Error;

/// Encoder errors.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EncodeError {
    /// The token does not have a head.
    MissingHead { token: usize, sent: Vec<String> },

    /// The token's head does not have a part-of-speech.
    MissingPOS { sent: Vec<String>, token: usize },

    /// The token does not have a dependency relation.
    MissingRelation { token: usize, sent: Vec<String> },
}

impl EncodeError {
    /// Construct `EncodeError::MissingHead` from a CoNLL-X graph.
    ///
    /// Construct an error. `token` is the node index for which the
    /// error applies in `sentence`.
    pub fn missing_head(token: usize, sentence: &Sentence) -> EncodeError {
        EncodeError::MissingHead {
            sent: Self::sentence_to_forms(sentence),
            token: token - 1,
        }
    }

    /// Construct `EncodeError::MissingPOS` from a CoNLL-X graph.
    ///
    /// Construct an error. `token` is the node index for which the
    /// error applies in `sentence`.
    pub fn missing_pos(token: usize, sentence: &Sentence) -> EncodeError {
        EncodeError::MissingPOS {
            sent: Self::sentence_to_forms(sentence),
            token: token - 1,
        }
    }

    /// Construct `EncodeError::MissingRelation` from a CoNLL-X graph.
    ///
    /// Construct an error. `token` is the node index for which the
    /// error applies in `sentence`.
    pub fn missing_relation(token: usize, sentence: &Sentence) -> EncodeError {
        EncodeError::MissingRelation {
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
            MissingPOS { token, sent } => write!(
                f,
                "Head of token '{}' does not have a part-of-speech:\n\n{}\n",
                sent[*token],
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

/// Decoder errors.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum DecodeError {
    /// The head position is out of bounds.
    #[error("position out of bounds")]
    PositionOutOfBounds,

    /// The head part-of-speech tag does not occur in the sentence.
    #[error("unknown part-of-speech tag")]
    InvalidPOS,
}
