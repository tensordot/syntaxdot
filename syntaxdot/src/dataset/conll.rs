use std::io::{BufRead, Seek, SeekFrom};

use conllu::io::{ReadSentence, Reader, Sentences};
use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};

use crate::dataset::DataSet;
use crate::error::SyntaxDotError;

/// A CoNLL-X data set.
pub struct ConlluDataSet<R>(R);

impl<R> ConlluDataSet<R> {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        ConlluDataSet(read)
    }
}

impl<'a, R> DataSet<'a> for &'a mut ConlluDataSet<R>
where
    R: BufRead + Seek,
{
    type Iter = ConllIter<'a, Reader<&'a mut R>>;

    fn sentences(self, tokenizer: &'a dyn Tokenize) -> Result<Self::Iter, SyntaxDotError> {
        // Rewind to the beginning of the dataset (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let reader = Reader::new(&mut self.0);

        Ok(ConllIter {
            sentences: reader.sentences(),
            tokenizer,
        })
    }
}

pub struct ConllIter<'a, R>
where
    R: ReadSentence,
{
    sentences: Sentences<R>,
    tokenizer: &'a dyn Tokenize,
}

impl<'a, R> Iterator for ConllIter<'a, R>
where
    R: ReadSentence,
{
    type Item = Result<SentenceWithPieces, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sentences.next().map(|s| {
            s.map(|s| self.tokenizer.tokenize(s))
                .map_err(SyntaxDotError::ConlluIOError)
        })
    }
}
