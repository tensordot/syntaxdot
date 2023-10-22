use std::io::{BufRead, Lines, Seek};

use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};
use udgraph::graph::Sentence;
use udgraph::token::Token;

use crate::dataset::{DataSet, PairedDataSet};
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
    type Iter = TokenizeIter<'a, &'a mut R>;

    fn tokenize(self, tokenizer: &'a dyn Tokenize) -> Result<Self::Iter, SyntaxDotError> {
        // Rewind to the beginning of the dataset (if necessary).
        self.0.rewind()?;

        Ok(TokenizeIter {
            lines: (&mut self.0).lines(),
            tokenizer,
        })
    }
}

impl<'a, R> PairedDataSet<'a> for &'a mut PlainTextDataSet<R>
where
    R: BufRead + Seek,
{
    type Iter = TokenizePairIter<'a, &'a mut R>;

    fn tokenize_pair(
        self,
        tokenizer1: &'a dyn Tokenize,
        tokenizer2: &'a dyn Tokenize,
    ) -> Result<Self::Iter, SyntaxDotError> {
        self.0.rewind()?;

        Ok(TokenizePairIter {
            lines: (&mut self.0).lines(),
            tokenizer1,
            tokenizer2,
        })
    }
}

pub struct TokenizeIter<'a, R> {
    lines: Lines<R>,
    tokenizer: &'a dyn Tokenize,
}

impl<'a, R> Iterator for TokenizeIter<'a, R>
where
    R: BufRead,
{
    type Item = Result<SentenceWithPieces, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        for line in &mut self.lines {
            // Bubble up read errors.
            let line = match line {
                Ok(line) => line,
                Err(err) => return Some(Err(SyntaxDotError::IoError(err))),
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

pub struct TokenizePairIter<'a, R> {
    lines: Lines<R>,
    tokenizer1: &'a dyn Tokenize,
    tokenizer2: &'a dyn Tokenize,
}

impl<'a, R> Iterator for TokenizePairIter<'a, R>
where
    R: BufRead,
{
    type Item = Result<(SentenceWithPieces, SentenceWithPieces), SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        for line in &mut self.lines {
            // Bubble up read errors.
            let line = match line {
                Ok(line) => line,
                Err(err) => return Some(Err(SyntaxDotError::IoError(err))),
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

            return Some(Ok((
                self.tokenizer1.tokenize(sentence.clone()),
                self.tokenizer2.tokenize(sentence),
            )));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Cursor};

    use crate::dataset::tests::{dataset_to_pieces, wordpiece_tokenizer, CORRECT_PIECE_IDS};
    use crate::dataset::PlainTextDataSet;

    const SENTENCES: &str = r#"
Dit is de eerste zin .
Dit de tweede zin .

En nu de laatste zin ."#;

    #[test]
    fn plain_text_dataset_works() {
        let tokenizer = wordpiece_tokenizer();
        let mut cursor = Cursor::new(SENTENCES);
        let mut dataset = PlainTextDataSet::new(BufReader::new(&mut cursor));

        let pieces = dataset_to_pieces(&mut dataset, &tokenizer).unwrap();
        assert_eq!(pieces, *CORRECT_PIECE_IDS);

        // Verify that the data set is correctly read again.
        let more_pieces = dataset_to_pieces(&mut dataset, &tokenizer).unwrap();
        assert_eq!(more_pieces, *CORRECT_PIECE_IDS);
    }
}
