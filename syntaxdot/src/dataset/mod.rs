//! Iterators over data sets.

use crate::encoders::NamedEncoder;
use crate::error::SyntaxDotError;
use crate::input::Tokenize;
use crate::tensor::Tensors;

pub(crate) mod tensor_iter;

pub use conll::ConlluDataSet;

mod conll;
pub(crate) mod sentence_iter;

/// A set of training/validation data.
///
/// A `DataSet` provides an iterator over the batches in that
/// dataset.
pub trait DataSet<'a> {
    type Iter: Iterator<Item = Result<Tensors, SyntaxDotError>>;

    /// Get an iterator over the dataset batches.
    ///
    /// The sequence inputs are encoded with the given `vectorizer`,
    /// the sequence labels using the `encoder`.
    ///
    /// Sentences longer than `max_len` are skipped. If you want to
    /// include all sentences, `max_len` should be `None`.
    ///
    /// If `shuffle_buffer_size` is not `None`, the given number of
    /// sentences will be accumulated and then shuffled.
    ///
    /// If `encoders` is not `None`, output tensors will be created
    /// for the labels in the data set.
    #[allow(clippy::too_many_arguments)]
    fn batches(
        self,
        tokenizer: &'a dyn Tokenize,
        encoders: Option<&'a [NamedEncoder]>,
        batch_size: usize,
        max_len: Option<SequenceLength>,
        shuffle_buffer_size: Option<usize>,
    ) -> Result<Self::Iter, SyntaxDotError>;
}

/// The length of a sequence.
///
/// This enum can be used to express the (maximum) length of a
/// sentence in tokens or in pieces.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SequenceLength {
    Tokens(usize),
    Pieces(usize),
}
