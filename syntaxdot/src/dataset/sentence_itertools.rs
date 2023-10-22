use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use syntaxdot_tokenizers::SentenceWithPieces;

use crate::error::SyntaxDotError;
use crate::util::RandomRemoveVec;

/// The maximum length of a sentence.
///
/// This enum can be used to express the (maximum) length of a
/// sentence in tokens or in pieces.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MaxSentenceLen {
    Tokens(usize),
    Pieces(usize),
    Unbounded,
}

/// Trait providing adapters for `SentenceWithPieces` iterators.
pub trait SentenceIterTools<'a>: Sized
where
    Self: Iterator,
{
    /// Filter sentences by their length.
    ///
    /// If `max_len` is `None`, then the sentences will not be
    /// filtered by length.
    fn filter_by_len(self, max_len: MaxSentenceLen) -> LengthFilter<Self>;

    /// Shuffle sentences.
    ///
    /// `buffer_size` is the size of the shuffle buffer that should be
    /// used. If `buffer_size` is `None`, then the sentences will not
    /// be shuffled.
    fn shuffle(self, buffer_size: usize) -> Shuffled<Self>;
}

impl<'a, I> SentenceIterTools<'a> for I
where
    I: 'a + Iterator,
{
    fn filter_by_len(self, max_len: MaxSentenceLen) -> LengthFilter<Self> {
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

/// Get the length of a tokenized sentence.
trait SentenceLen {
    /// Get the length of the sentence in pieces.
    fn pieces_len(&self) -> usize;

    /// Get the length of the sentence in tokens.
    fn tokens_len(&self) -> usize;
}

impl SentenceLen for SentenceWithPieces {
    fn pieces_len(&self) -> usize {
        self.pieces.len()
    }

    fn tokens_len(&self) -> usize {
        self.token_offsets.len()
    }
}

impl SentenceLen for (SentenceWithPieces, SentenceWithPieces) {
    fn pieces_len(&self) -> usize {
        self.0.pieces.len().max(self.1.pieces.len())
    }

    fn tokens_len(&self) -> usize {
        self.0.token_offsets.len().max(self.1.token_offsets.len())
    }
}

/// An Iterator adapter filtering sentences by maximum length.
pub struct LengthFilter<I> {
    inner: I,
    max_len: MaxSentenceLen,
}

impl<I, S> Iterator for LengthFilter<I>
where
    I: Iterator<Item = Result<S, SyntaxDotError>>,
    S: SentenceLen,
{
    type Item = Result<S, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        for sent in &mut self.inner {
            // Treat Err as length 0 to keep our type as Result<Sentence, Error>. The iterator
            // will properly return the Error at a later point.
            let too_long = match self.max_len {
                MaxSentenceLen::Pieces(max_len) => {
                    sent.as_ref().map(|s| s.pieces_len()).unwrap_or(0) > max_len
                }
                MaxSentenceLen::Tokens(max_len) => {
                    sent.as_ref().map(|s| s.tokens_len()).unwrap_or(0) > max_len
                }
                MaxSentenceLen::Unbounded => false,
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
pub struct Shuffled<I>
where
    I: Iterator,
{
    inner: I,
    buffer: RandomRemoveVec<I::Item, XorShiftRng>,
    buffer_size: usize,
}

impl<I, V> Iterator for Shuffled<I>
where
    I: Iterator<Item = V>,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            for sent in &mut self.inner {
                self.buffer.push(sent);

                if self.buffer.len() == self.buffer_size {
                    break;
                }
            }
        }

        match self.inner.next() {
            Some(sent) => Some(self.buffer.push_and_remove_random(sent)),
            None => self.buffer.remove_random(),
        }
    }
}
