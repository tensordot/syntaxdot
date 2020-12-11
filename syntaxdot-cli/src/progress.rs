use std::io::{self, Read, Seek, SeekFrom};
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};

const NANOS_PER_SEC: u32 = 1_000_000_000;

pub struct ReadProgress<R> {
    inner: R,
    progress_bar: ProgressBar,
}

/// A progress bar that implements the `Read` and `Seek` traits.
///
/// This wrapper of `indicatif`'s `ProgressBar` updates progress based on the
/// current offset within the file.
impl<R> ReadProgress<R>
where
    R: Seek,
{
    pub fn new(mut read: R) -> io::Result<Self> {
        let len = read.seek(SeekFrom::End(0))? + 1;
        read.seek(SeekFrom::Start(0))?;
        let progress_bar = ProgressBar::new(len);
        progress_bar
            .set_style(ProgressStyle::default_bar().template("{bar} {bytes}/{total_bytes}"));

        Ok(ReadProgress {
            inner: read,
            progress_bar,
        })
    }

    pub fn progress_bar(&self) -> &ProgressBar {
        &self.progress_bar
    }
}

impl<R> Read for ReadProgress<R>
where
    R: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n_read = self.inner.read(buf)?;
        let pos = self.inner.seek(SeekFrom::Current(0))?;
        self.progress_bar.set_position(pos);
        Ok(n_read)
    }
}

impl<R> Seek for ReadProgress<R>
where
    R: Seek,
{
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let pos = self.inner.seek(pos)?;
        self.progress_bar.set_position(pos);
        Ok(pos)
    }
}

impl<R> Drop for ReadProgress<R> {
    fn drop(&mut self) {
        self.progress_bar.finish();
    }
}

/// Measure the number of sentences processed per second.
///
/// When an instance of `TaggerSpeed` is constructed, it takes the
/// current time. `count_sentence` should be called for each sentence
/// that was processed. A `TaggerSpeed` instance will print the
/// processing speed to *stderr* when the instance is dropped.
pub struct TaggerSpeed {
    start: Instant,
    n_sentences: usize,
}

impl TaggerSpeed {
    /// Construct a new instance.
    pub fn new() -> Self {
        TaggerSpeed {
            start: Instant::now(),
            n_sentences: 0,
        }
    }

    /// Count a processed sentences.
    pub fn count_sentence(&mut self) {
        self.n_sentences += 1;
    }
}

impl Default for TaggerSpeed {
    fn default() -> Self {
        TaggerSpeed::new()
    }
}

impl Drop for TaggerSpeed {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        // From nightly-only as_secs_f32.
        let elapsed_secs =
            (elapsed.as_secs() as f32) + (elapsed.subsec_nanos() as f32) / (NANOS_PER_SEC as f32);
        log::info!(
            "Processed {} sentences in {:.1}s ({:.1} sents/s)",
            self.n_sentences,
            elapsed_secs,
            self.n_sentences as f32 / elapsed_secs
        );
    }
}
