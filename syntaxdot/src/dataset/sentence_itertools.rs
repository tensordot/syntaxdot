use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use syntaxdot_tokenizers::SentenceWithPieces;

use crate::error::SyntaxDotError;
use crate::util::RandomRemoveVec;

/// The length of a sequence.
///
/// This enum can be used to express the (maximum) length of a
/// sentence in tokens or in pieces.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SequenceLength {
    Tokens(usize),
    Pieces(usize),
    Unbounded,
}

/// Trait providing adapters for `SentenceWithPieces` iterators.
pub trait SentenceIterTools<'a>: Sized {
    /// Filter sentences by their length.
    ///
    /// If `max_len` is `None`, then the sentences will not be
    /// filtered by length.
    fn filter_by_len(self, max_len: SequenceLength) -> LengthFilter<Self>;

    /// Shuffle sentences.
    ///
    /// `buffer_size` is the size of the shuffle buffer that should be
    /// used. If `buffer_size` is `None`, then the sentences will not
    /// be shuffled.
    fn shuffle(self, buffer_size: usize) -> Shuffled<Self>;
}

impl<'a, I> SentenceIterTools<'a> for I
where
    I: 'a + Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
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
        for sent in &mut self.inner {
            // Treat Err as length 0 to keep our type as Result<Sentence, Error>. The iterator
            // will properly return the Error at a later point.
            let too_long = match self.max_len {
                SequenceLength::Pieces(max_len) => {
                    sent.as_ref().map(|s| s.pieces.len()).unwrap_or(0) > max_len
                }
                SequenceLength::Tokens(max_len) => {
                    sent.as_ref().map(|s| s.token_offsets.len()).unwrap_or(0) > max_len
                }
                SequenceLength::Unbounded => false,
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
            for sent in &mut self.inner {
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
