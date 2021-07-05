use std::iter::Peekable;
use std::ops::Deref;

use anyhow::Result;
use conllu::io::WriteSentence;
use rayon::iter::{ParallelBridge, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::cmp;
use syntaxdot::tagger::Tagger;
use syntaxdot_tokenizers::SentenceWithPieces;

struct TaggerWrap<'a>(&'a Tagger);

unsafe impl<'a> Send for TaggerWrap<'a> {}

unsafe impl<'a> Sync for TaggerWrap<'a> {}

impl<'a> Deref for TaggerWrap<'a> {
    type Target = Tagger;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SentProcessor<'a, W>
where
    W: WriteSentence,
{
    tagger: TaggerWrap<'a>,
    writer: W,
    max_batch_pieces: usize,
    max_len: Option<usize>,
    read_ahead: usize,
    buffer: Vec<SentenceWithPieces>,
}

impl<'a, W> SentProcessor<'a, W>
where
    W: WriteSentence,
{
    /// Construct a new sentence processor.
    ///
    /// The sentence processor uses `tokenizer` and `tagger` to
    /// process a sentence. The annotated sentences are written to
    /// `writer`. The sentences are batched, so that batches contain at
    /// most `max_batch_pieces` pieces. Sentences that are longer than
    /// `max_len` are ignored. The processor reads ahead `read_ahead`
    /// sentences before starting to process sentences. This read-ahead
    /// is used to sort sentences by length to speed up processing.
    pub fn new(
        tagger: &'a Tagger,
        writer: W,
        max_batch_pieces: usize,
        max_len: Option<usize>,
        read_ahead: usize,
    ) -> Self {
        assert!(
            max_batch_pieces > 0,
            "Maximum batch pieces should at least be 1."
        );
        assert!(read_ahead > 0, "Read ahead should at least be 1.");

        SentProcessor {
            tagger: TaggerWrap(tagger),
            writer,
            max_batch_pieces,
            max_len,
            read_ahead,
            buffer: Vec::with_capacity(read_ahead),
        }
    }

    /// Process a sentence.
    ///
    /// The sentences are not annotated until `read_ahead` sentences
    /// are queued using this method or the destructor is invoked. Once one of these
    /// two conditions are met, the sentences are annotated and written after
    /// annotation.
    ///
    /// The annotation of sentences is parallelized using Rayon. By default, the
    /// global Rayon thread pool is used. Another [`rayon::ThreadPool`] can be used
    /// through [`rayon::ThreadPool::install`], however then `SentenceProcessor`
    /// must also be destructed in the scope of the closure.
    pub fn process(&mut self, tokenized_sentence: SentenceWithPieces) -> Result<()> {
        if let Some(max_len) = self.max_len {
            // sent.len() includes the root node, whereas max_len is
            // the actual sentence length without the root node.
            if (tokenized_sentence.pieces.len() - 1) > max_len {
                return Ok(());
            }
        }

        self.buffer.push(tokenized_sentence);

        if self.buffer.len() == self.read_ahead {
            self.tag_buffered_sentences()?;
        }

        Ok(())
    }

    fn tag_buffered_sentences(&mut self) -> Result<()> {
        // Sort sentences by length.
        let mut sent_refs: Vec<_> = self.buffer.iter_mut().collect();
        sent_refs.par_sort_unstable_by_key(|s| s.pieces.len());

        // Convince the type system that we are not borrowing SentProcessor, which is
        // not Sync.
        let tagger = &self.tagger;

        // Split in batches, tag, and merge results.
        sent_refs
            .into_iter()
            .max_pieces_batches(self.max_batch_pieces)
            .par_bridge()
            .try_for_each(|mut batch| tagger.tag_sentences(&mut batch))?;

        // Write out sentences.
        let mut sents = Vec::with_capacity(self.read_ahead);
        std::mem::swap(&mut sents, &mut self.buffer);
        for sent in sents {
            self.writer.write_sentence(&sent.sentence)?;
        }

        Ok(())
    }
}

impl<'a, W> Drop for SentProcessor<'a, W>
where
    W: WriteSentence,
{
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            if let Err(err) = self.tag_buffered_sentences() {
                log::error!("Error tagging sentences: {}", err);
            }
        }
    }
}

trait MaxPieces<'a> {
    fn max_pieces_batches(self, max_batch_pieces: usize) -> MaxPiecesIter<'a, Self>
    where
        Self: Sized + Iterator<Item = &'a mut SentenceWithPieces>;
}

impl<'a, I> MaxPieces<'a> for I
where
    I: Iterator<Item = &'a mut SentenceWithPieces>,
{
    fn max_pieces_batches(self, max_batch_pieces: usize) -> MaxPiecesIter<'a, Self> {
        MaxPiecesIter {
            inner: self.peekable(),
            max_batch_pieces,
        }
    }
}

struct MaxPiecesIter<'a, I>
where
    I: Iterator<Item = &'a mut SentenceWithPieces>,
{
    inner: Peekable<I>,
    max_batch_pieces: usize,
}

impl<'a, I> Iterator for MaxPiecesIter<'a, I>
where
    I: Iterator<Item = &'a mut SentenceWithPieces>,
{
    type Item = Vec<&'a mut SentenceWithPieces>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::new();
        let mut max_seq_len = 0;

        loop {
            match self.inner.peek() {
                Some(next) => {
                    // Compute how many pieces the batch would contain if the next sentence is added.
                    let n_pieces = (batch.len() + 1) * cmp::max(max_seq_len, next.pieces.len());

                    // If adding the next sentence would cross the threshold, return the batch as-is,
                    // unless the batch is empty.
                    if n_pieces > self.max_batch_pieces && !batch.is_empty() {
                        return Some(batch);
                    }
                }
                None => {
                    // If there are no more sentences, return the current batch if it is empty,
                    // or `None` otherwise to signal that the iterator is exhausted.
                    if batch.is_empty() {
                        return None;
                    } else {
                        return Some(batch);
                    }
                }
            }

            // Unwrapping is safe, since we already peeked above.
            let next = self.inner.next().unwrap();

            max_seq_len = cmp::max(max_seq_len, next.pieces.len());
            batch.push(next);
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use syntaxdot_tokenizers::SentenceWithPieces;

    use crate::sent_proc::MaxPieces;

    #[test]
    fn max_pieces_iter_handles_empty() {
        let mut sentences = vec![];
        let mut iter = sentences.iter_mut().max_pieces_batches(4);
        assert_eq!(iter.next(), None as Option<Vec<_>>);
    }

    #[test]
    fn max_pieces_iter_handles_longer_than_max_batch_pieces() {
        let mut sentences = vec![SentenceWithPieces {
            pieces: array![0, 1, 2, 3, 4, 5],
            sentence: Default::default(),
            token_offsets: vec![],
        }];

        let mut iter = sentences.iter_mut().max_pieces_batches(4);

        assert_eq!(
            iter.next(),
            Some(vec![&mut SentenceWithPieces {
                pieces: array![0, 1, 2, 3, 4, 5],
                sentence: Default::default(),
                token_offsets: vec![]
            },])
        );

        assert_eq!(iter.next(), None as Option<Vec<_>>);
    }

    #[test]
    fn max_pieces_iter_splits_in_batches() {
        let mut sentences = vec![
            SentenceWithPieces {
                pieces: array![0, 1],
                sentence: Default::default(),
                token_offsets: vec![],
            },
            SentenceWithPieces {
                pieces: array![2, 3],
                sentence: Default::default(),
                token_offsets: vec![],
            },
            SentenceWithPieces {
                pieces: array![4, 5, 6],
                sentence: Default::default(),
                token_offsets: vec![],
            },
            SentenceWithPieces {
                pieces: array![7, 8],
                sentence: Default::default(),
                token_offsets: vec![],
            },
        ];

        let mut iter = sentences.iter_mut().max_pieces_batches(4);

        assert_eq!(
            iter.next(),
            Some(vec![
                &mut SentenceWithPieces {
                    pieces: array![0, 1],
                    sentence: Default::default(),
                    token_offsets: vec![]
                },
                &mut SentenceWithPieces {
                    pieces: array![2, 3],
                    sentence: Default::default(),
                    token_offsets: vec![]
                }
            ])
        );

        assert_eq!(
            iter.next(),
            Some(vec![&mut SentenceWithPieces {
                pieces: array![4, 5, 6],
                sentence: Default::default(),
                token_offsets: vec![]
            }])
        );

        assert_eq!(
            iter.next(),
            Some(vec![&mut SentenceWithPieces {
                pieces: array![7, 8],
                sentence: Default::default(),
                token_offsets: vec![]
            }])
        );

        assert_eq!(iter.next(), None as Option<Vec<_>>);
    }
}
