use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use syntaxdot_tokenizers::SentenceWithPieces;

use crate::dataset::SequenceLength;
use crate::error::SyntaxDotError;
use crate::util::RandomRemoveVec;

/// Trait providing adapters for `SentenceWithPieces` iterators.
pub trait SentenceIter: Sized {
    /// Filter sentences by their length.
    fn filter_by_len(self, max_len: SequenceLength) -> LengthFilter<Self>;

    /// Shuffle sentences.
    ///
    /// `buffer_size` is the size of the shuffle buffer that should be used.
    fn shuffle(self, buffer_size: usize) -> Shuffled<Self>;
}

impl<I> SentenceIter for I
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    fn filter_by_len(self, max_len: SequenceLength) -> LengthFilter<Self> {
        LengthFilter {
            inner: self,
            max_len,
        }
    }

    fn shuffle(self, buffer_size: usize) -> Shuffled<Self> {
        Shuffled {
            inner: self,
            buffer: RandomRemoveVec::with_capacity(buffer_size, XorShiftRng::from_entropy()),
            buffer_size,
        }
    }
}

/// An Iterator adapter filtering sentences by maximum length.
pub struct LengthFilter<I> {
    inner: I,
    max_len: SequenceLength,
}

impl<I> Iterator for LengthFilter<I>
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    type Item = Result<SentenceWithPieces, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(sent) = self.inner.next() {
            // Treat Err as length 0 to keep our type as Result<Sentence, Error>. The iterator
            // will properly return the Error at a later point.
            let too_long = match self.max_len {
                SequenceLength::Pieces(max_len) => {
                    sent.as_ref().map(|s| s.pieces.len()).unwrap_or(0) > max_len
                }
                SequenceLength::Tokens(max_len) => {
                    sent.as_ref().map(|s| s.token_offsets.len()).unwrap_or(0) > max_len
                }
            };

            if too_long {
                continue;
            }

            return Some(sent);
        }
        None
    }
}

/// An Iterator adapter performing local shuffling.
///
/// Fills a buffer with size `buffer_size` on the first
/// call. Subsequent calls add the next incoming item to the buffer
/// and pick a random element from the buffer.
pub struct Shuffled<I> {
    inner: I,
    buffer: RandomRemoveVec<SentenceWithPieces, XorShiftRng>,
    buffer_size: usize,
}

impl<I> Iterator for Shuffled<I>
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    type Item = Result<SentenceWithPieces, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            while let Some(sent) = self.inner.next() {
                match sent {
                    Ok(sent) => self.buffer.push(sent),
                    Err(err) => return Some(Err(err)),
                }

                if self.buffer.len() == self.buffer_size {
                    break;
                }
            }
        }

        match self.inner.next() {
            Some(sent) => match sent {
                Ok(sent) => Some(Ok(self.buffer.push_and_remove_random(sent))),
                Err(err) => Some(Err(err)),
            },
            None => self.buffer.remove_random().map(Result::Ok),
        }
    }
}
