//! Iterators over data sets.

use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};

use crate::error::SyntaxDotError;

mod conll;
pub use conll::ConlluDataSet;

mod plaintext;
pub use plaintext::PlainTextDataSet;

pub(crate) mod tensor_iter;
pub use tensor_iter::{BatchedTensors, PairedBatchedTensors};

mod sentence_itertools;
pub use sentence_itertools::{MaxSentenceLen, SentenceIterTools};

/// A data set consisting of annotated or unannotated sentences.
///
/// A `DataSet` provides an iterator over the sentences (and their
/// pieces) in a data set.
pub trait DataSet<'a> {
    type Iter: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>;

    /// Get an iterator over the sentences and pieces in a dataset.
    ///
    /// The tokens are split in pieces with the given `tokenizer`.
    fn tokenize(self, tokenizer: &'a dyn Tokenize) -> Result<Self::Iter, SyntaxDotError>;
}

/// A data set consisting of annotated or unannotated sentences.
///
/// A `DataSet` provides an iterator over pairs of tokenized sentences.
/// The pairing is useful in applications where two tokenizer models
/// are used, such as distillation.
pub trait PairedDataSet<'a> {
    type Iter: Iterator<Item = Result<(SentenceWithPieces, SentenceWithPieces), SyntaxDotError>>;

    /// Get an iterator over the sentences and pieces in a dataset.
    ///
    /// The tokens are split in pieces with the given tokenizers.
    fn tokenize_pair(
        self,
        tokenizer1: &'a dyn Tokenize,
        tokenizer2: &'a dyn Tokenize,
    ) -> Result<Self::Iter, SyntaxDotError>;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::{BufReader, Cursor};

    use lazy_static::lazy_static;
    use ndarray::{array, Array1};
    use syntaxdot_tokenizers::{BertTokenizer, SentenceWithPieces, Tokenize};
    use wordpieces::WordPieces;

    use crate::dataset::DataSet;
    use crate::error::SyntaxDotError;

    const PIECES: &str = r#"[CLS]
[UNK]
Dit
is
de
eerste
zin
.
tweede
laatste
nu"#;

    lazy_static! {
        pub static ref CORRECT_PIECE_IDS: Vec<Array1<i64>> = vec![
            array![0, 2, 3, 4, 5, 6, 7],
            array![0, 2, 4, 8, 6, 7],
            array![0, 1, 10, 4, 9, 6, 7]
        ];
    }

    pub fn dataset_to_pieces<'a, D, I>(
        dataset: D,
        tokenizer: &'a dyn Tokenize,
    ) -> Result<Vec<Array1<i64>>, SyntaxDotError>
    where
        D: DataSet<'a, Iter = I>,
        I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
    {
        dataset
            .tokenize(tokenizer)?
            .map(|s| s.map(|s| s.pieces))
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn wordpiece_tokenizer() -> BertTokenizer {
        let pieces = WordPieces::from_buf_read(BufReader::new(Cursor::new(PIECES))).unwrap();
        BertTokenizer::new(pieces, "[UNK]")
    }
}
