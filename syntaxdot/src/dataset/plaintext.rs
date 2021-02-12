use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};
use udgraph::token::Token;

use crate::dataset::sentence_iter::SentenceIter;
use crate::dataset::tensor_iter::TensorIter;
use crate::dataset::{DataSet, SequenceLength};
use crate::encoders::NamedEncoder;
use crate::error::SyntaxDotError;

/// A CoNLL-X data set.
pub struct PlainTextDataSet<R>(R);

impl<R> PlainTextDataSet<R> {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        Self(read)
    }

    /// Returns an `Iterator` over `Result<Sentence, Error>`.
    ///
    /// Depending on the parameters the returned iterator filters
    /// sentences by their lengths or returns the sentences in
    /// sequence without filtering them.
    ///
    /// If `max_len` == `None`, no filtering is performed.
    fn get_sentence_iter<'a>(
        reader: R,
        tokenizer: &'a dyn Tokenize,
    ) -> impl 'a + Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>
    where
        R: BufRead + 'a,
    {
        reader
            .lines()
            // Filter empty lines.
            .filter(|line| line.as_ref().map(|l| !l.is_empty()).unwrap_or(true))
            // Split lines into tokens.
            .map(|line| {
                line.map(|l| {
                    l.trim()
                        .split_terminator(' ')
                        .map(ToString::to_string)
                        .map(Token::new)
                        .collect()
                })
            })
            // Apply piece tokenization.
            .map(move |s| s.map(|s| tokenizer.tokenize(s)))
            // Convert errors.
            .map(|s| s.map_err(SyntaxDotError::IOError))
    }
}

impl<'a, R> DataSet<'a> for &'a mut PlainTextDataSet<R>
where
    R: Read + Seek,
{
    type Iter =
        TensorIter<'a, Box<dyn Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>> + 'a>>;

    fn batches(
        self,
        tokenizer: &'a dyn Tokenize,
        biaffine_encoder: Option<&'a ImmutableDependencyEncoder>,
        encoders: Option<&'a [NamedEncoder]>,
        batch_size: usize,
        max_len: Option<SequenceLength>,
        shuffle_buffer_size: Option<usize>,
    ) -> Result<Self::Iter, SyntaxDotError> {
        // Rewind to the beginning of the dataset (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let sentence_iter =
            PlainTextDataSet::get_sentence_iter(BufReader::new(&mut self.0), tokenizer);

        let sentences: Box<dyn Iterator<Item = _>> = match (max_len, shuffle_buffer_size) {
            (Some(max_len), None) => Box::new(sentence_iter.filter_by_len(max_len)),
            (None, Some(shuffle_buffer_size)) => {
                Box::new(sentence_iter.shuffle(shuffle_buffer_size))
            }
            (Some(max_len), Some(shuffle_buffer_size)) => Box::new(
                sentence_iter
                    .filter_by_len(max_len)
                    .shuffle(shuffle_buffer_size),
            ),
            (None, None) => Box::new(sentence_iter),
        };

        Ok(TensorIter {
            batch_size,
            biaffine_encoder,
            encoders,
            sentences,
        })
    }
}
