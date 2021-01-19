use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufRead, BufReader};

use conllu::graph::{Node, Sentence};
use wordpieces::WordPieces;

use super::{SentenceWithPieces, Tokenize};
use crate::TokenizerError;
use std::path::Path;

/// BERT word piece tokenizer.
///
/// This tokenizer splits CoNLL-X tokens into word pieces. For
/// example, a sentence such as:
///
/// > Veruntreute die AWO Spendengeld ?
///
/// Could be split (depending on the vocabulary) into the following
/// word pieces:
///
/// > Ver ##unt ##reute die A ##W ##O Spenden ##geld [UNK]
///
/// Then vocabulary index of each such piece is returned.
///
/// The unknown token (here `[UNK]`) can be specified while
/// constructing a tokenizer.
pub struct BertTokenizer {
    word_pieces: WordPieces,
    unknown_piece: String,
}

impl BertTokenizer {
    /// Construct a tokenizer from wordpieces and the unknown piece.
    pub fn new(word_pieces: WordPieces, unknown_piece: impl Into<String>) -> Self {
        BertTokenizer {
            word_pieces,
            unknown_piece: unknown_piece.into(),
        }
    }

    pub fn open<P>(model_path: P, unknown_piece: impl Into<String>) -> Result<Self, TokenizerError>
    where
        P: AsRef<Path>,
    {
        let model_path = model_path.as_ref();
        let f = File::open(model_path)
            .map_err(|err| TokenizerError::open_error(model_path.to_string_lossy(), err))?;
        Self::read(BufReader::new(f), unknown_piece)
    }

    pub fn read<R>(
        buf_read: R,
        unknown_piece: impl Into<String>,
    ) -> Result<BertTokenizer, TokenizerError>
    where
        R: BufRead,
    {
        let word_pieces = WordPieces::try_from(buf_read.lines())?;
        Ok(Self::new(word_pieces, unknown_piece))
    }
}

impl Tokenize for BertTokenizer {
    fn tokenize(&self, sentence: Sentence) -> SentenceWithPieces {
        // An average of three pieces per token ought to enough for
        // everyone ;).
        let mut pieces = Vec::with_capacity((sentence.len() - 1) * 3);
        let mut token_offsets = Vec::with_capacity(sentence.len());

        pieces.push(
            self.word_pieces
                .get_initial("[CLS]")
                .expect("BERT model does not have a [CLS] token") as i64,
        );

        for token in sentence.iter().filter_map(Node::token) {
            token_offsets.push(pieces.len());

            match self
                .word_pieces
                .split(token.form())
                .map(|piece| piece.idx().map(|piece| piece as i64))
                .collect::<Option<Vec<_>>>()
            {
                Some(word_pieces) => pieces.extend(word_pieces),
                None => pieces.push(
                    self.word_pieces
                        .get_initial(&self.unknown_piece)
                        .expect("Cannot get unknown piece") as i64,
                ),
            }
        }

        SentenceWithPieces {
            pieces: pieces.into(),
            sentence,
            token_offsets,
        }
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::iter::FromIterator;

    use conllu::graph::Sentence;
    use conllu::token::Token;
    use ndarray::array;
    use wordpieces::WordPieces;

    use super::BertTokenizer;
    use crate::Tokenize;

    fn read_pieces() -> WordPieces {
        let f = File::open(env!("BERT_BASE_GERMAN_CASED_VOCAB")).unwrap();
        WordPieces::try_from(BufReader::new(f).lines()).unwrap()
    }

    fn sentence_from_forms(forms: &[&str]) -> Sentence {
        Sentence::from_iter(forms.iter().map(|&f| Token::new(f)))
    }

    #[test]
    fn test_pieces() {
        let tokenizer = BertTokenizer::new(read_pieces(), "[UNK]");

        let sentence = sentence_from_forms(&["Veruntreute", "die", "AWO", "Spendengeld", "?"]);

        let sentence_pieces = tokenizer.tokenize(sentence);
        assert_eq!(
            sentence_pieces.pieces,
            array![3i64, 133, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2]
        );
        assert_eq!(sentence_pieces.token_offsets, &[1, 4, 5, 8, 10]);
    }
}
