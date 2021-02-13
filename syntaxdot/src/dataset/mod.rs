//! Iterators over data sets.

use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};

use crate::error::SyntaxDotError;

mod conll;
pub use conll::ConlluDataSet;

mod plaintext;
pub use plaintext::PlainTextDataSet;

pub(crate) mod tensor_iter;
pub use tensor_iter::BatchedTensors;

mod sentence_itertools;
pub use sentence_itertools::{SentenceIterTools, SequenceLength};

/// A data set consisting of annotated or unannotated sentences.
///
/// A `DataSet` provides an iterator over the sentences (and their
/// pieces) in a data set.
pub trait DataSet<'a> {
    type Iter: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>;

    /// Get an iterator over the sentences and pieces in a dataset.
    ///
    /// The tokens are split in pieces with the given `tokenizer`.
    fn sentences(self, tokenizer: &'a dyn Tokenize) -> Result<Self::Iter, SyntaxDotError>;
}
