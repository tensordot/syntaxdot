use std::io::{BufRead, Lines, Seek, SeekFrom};

use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};
use udgraph::graph::Sentence;
use udgraph::token::Token;

use crate::dataset::DataSet;
use crate::error::SyntaxDotError;

/// A CoNLL-X data set.
pub struct PlainTextDataSet<R>(R);

impl<R> PlainTextDataSet<R> {
    /// Construct a plain-text dataset.
    pub fn new(read: R) -> Self {
        Self(read)
    }
}

impl<'a, R> DataSet<'a> for &'a mut PlainTextDataSet<R>
where
    R: BufRead + Seek,
{
    type Iter = PlainTextIter<'a, &'a mut R>;

    fn sentences(self, tokenizer: &'a dyn Tokenize) -> Result<Self::Iter, SyntaxDotError> {
        // Rewind to the beginning of the dataset (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        Ok(PlainTextIter {
            lines: (&mut self.0).lines(),
            tokenizer,
        })
    }
}

pub struct PlainTextIter<'a, R> {
    lines: Lines<R>,
    tokenizer: &'a dyn Tokenize,
}

impl<'a, R> Iterator for PlainTextIter<'a, R>
where
    R: BufRead,
{
    type Item = Result<SentenceWithPieces, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(line) = self.lines.next() {
            // Bubble up read errors.
            let line = match line {
                Ok(line) => line,
                Err(err) => return Some(Err(SyntaxDotError::IOError(err))),
            };

            let line_trimmed = line.trim();

            // Skip empty lines
            if line_trimmed.is_empty() {
                continue;
            }

            let sentence = line_trimmed
                .split_terminator(' ')
                .map(ToString::to_string)
                .map(Token::new)
                .collect::<Sentence>();

            return Some(Ok(self.tokenizer.tokenize(sentence)));
        }

        None
    }
}
