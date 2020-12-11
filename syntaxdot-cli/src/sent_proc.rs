use anyhow::Result;
use conllu::graph::Sentence;
use conllu::io::WriteSentence;

use syntaxdot::tagger::Tagger;
use syntaxdot_tokenizers::{SentenceWithPieces, Tokenize};

pub struct SentProcessor<'a, W>
where
    W: WriteSentence,
{
    tokenizer: &'a dyn Tokenize,
    tagger: &'a Tagger,
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
        tokenizer: &'a dyn Tokenize,
        tagger: &'a Tagger,
        writer: W,
        batch_size: usize,
        max_len: Option<usize>,
        read_ahead: usize,
    ) -> Self {
        assert!(batch_size > 0, "Batch size should at least be 1.");
        assert!(read_ahead > 0, "Read ahead should at least be 1.");

        SentProcessor {
            tokenizer,
            tagger,
            writer,
            batch_size,
            max_len,
            read_ahead,
            buffer: Vec::with_capacity(read_ahead * batch_size),
        }
    }

    /// Process a sentence.
    pub fn process(&mut self, sent: Sentence) -> Result<()> {
        let tokenized_sentence = self.tokenizer.tokenize(sent);

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

        // Split in batches, tag, and merge results.
        for batch in sent_refs.chunks_mut(self.batch_size) {
            self.tagger.tag_sentences(batch)?;
        }

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
