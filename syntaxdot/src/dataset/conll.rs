use conllu::io::{ReadSentence, Reader};
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::dataset::sentence_iter::SentenceIter;
use crate::dataset::tensor_iter::TensorIter;
use crate::dataset::{DataSet, SequenceLength};
use crate::encoders::NamedEncoder;
use crate::error::SyntaxDotError;
use crate::input::{SentenceWithPieces, Tokenize};

/// A CoNLL-X data set.
pub struct ConlluDataSet<R>(R);

impl<R> ConlluDataSet<R> {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        ConlluDataSet(read)
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
    ) -> impl 'a + Iterator<Item = Result<SentenceWithPieces, conllu::IOError>>
    where
        R: ReadSentence + 'a,
    {
        reader
            .sentences()
            .map(move |s| s.map(|s| tokenizer.tokenize(s)))
    }
}

impl<'a, R> DataSet<'a> for &'a mut ConlluDataSet<R>
where
    R: Read + Seek,
{
    type Iter =
        TensorIter<'a, Box<dyn Iterator<Item = Result<SentenceWithPieces, conllu::IOError>> + 'a>>;

    fn batches(
        self,
        tokenizer: &'a dyn Tokenize,
        encoders: Option<&'a [NamedEncoder]>,
        batch_size: usize,
        max_len: Option<SequenceLength>,
        shuffle_buffer_size: Option<usize>,
    ) -> Result<Self::Iter, SyntaxDotError> {
        // Rewind to the beginning of the dataset (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let reader = Reader::new(BufReader::new(&mut self.0));

        let sentence_iter = ConlluDataSet::get_sentence_iter(reader, tokenizer);

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
            encoders,
            sentences,
        })
    }
}
