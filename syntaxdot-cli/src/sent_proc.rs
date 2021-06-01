use std::ops::Deref;

use anyhow::Result;
use conllu::io::WriteSentence;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
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
    batch_size: usize,
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
    /// `writer`. `batch_size` sentences are processed together,
    /// ignoring sentences that are longer than `max_len`. The
    /// processor reads ahead `read_ahead` batches before starting to
    /// process sentences. This read-ahead is used to sort sentences
    /// by length to speed up processing.
    pub fn new(
        tagger: &'a Tagger,
        writer: W,
        batch_size: usize,
        max_len: Option<usize>,
        read_ahead: usize,
    ) -> Self {
        assert!(batch_size > 0, "Batch size should at least be 1.");
        assert!(read_ahead > 0, "Read ahead should at least be 1.");

        SentProcessor {
            tagger: TaggerWrap(tagger),
            writer,
            batch_size,
            max_len,
            read_ahead,
            buffer: Vec::with_capacity(read_ahead * batch_size),
        }
    }

    /// Process a sentence.
    ///
    /// The sentences are not annotated until `batch_size * read_ahead` sentences
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

        if self.buffer.len() == self.batch_size * self.read_ahead {
            self.tag_buffered_sentences()?;
        }

        Ok(())
    }

    fn tag_buffered_sentences(&mut self) -> Result<()> {
        // Sort sentences by length.
        let mut sent_refs: Vec<_> = self.buffer.iter_mut().collect();
        sent_refs.sort_unstable_by_key(|s| s.pieces.len());

        // Convince the type system that we are not borrowing SentProcessor, which is
        // not Sync.
        let tagger = &self.tagger;

        // Split in batches, tag, and merge results.
        sent_refs
            .par_chunks_mut(self.batch_size)
            .try_for_each(|batch| tagger.tag_sentences(batch))?;

        // Write out sentences.
        let mut sents = Vec::with_capacity(self.read_ahead * self.batch_size);
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
