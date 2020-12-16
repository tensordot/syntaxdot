//! TensorBoard summary writer
//!
//! This crate implements just enough functionality to write scalars
//! in the TensorFlow/TensorBoard summary format. This replaces the
//! far more extensive `tfrecord` crate, which has **many**
//! dependencies.

mod crc32;

mod crc32_table;

pub(crate) mod event_writer;

pub(crate) mod record_writer;

mod summary_writer;
pub use summary_writer::SummaryWriter;
