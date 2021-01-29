# Changelog

## [Unreleased]

### Added

- Added support for biaffine dependency parsing (Dozat & Manning, 2016).
  Biaffine parsing is enabled through the `biaffine` configuration option.

### Changed

- Pretrained models are now loaded from the libtorch OutputArchive format,
  rather than the HDF5 format. This removes HDF5 as a dependency.
- Properly prefix embeddings with `embeddings` rather than `encoder` in
  BERT/RoBERTa models. **Warning:** This breaks compatibility with BERT and
  RoBERTa models from prior versions of SyntaxDot and sticker2, which should
  be retrained.
- Implementations of `Tokenizer` are now required to put a piece that marks the
  beginning of a sentence before the first token piece. `BertTokenizer` was the
  only tokenizer that did not fulfill this requirement. `BertTokenizer` is
  updated to insert the `[CLS]` piece as a beginning of sentence marker.
  **Warning:** this breaks existing models with `tokenizer = "bert"`, which should
  be retrained.
- Replace all calls to the Rust Torch crate (`tch`) by fallible counterparts,
  this makes exceptions thrown by Torch far easier to read.
- Uses of the `eprintln!` macro are replaced by logging using `log` and
  `env_logger`. The verbosity of the logs can be controlled with the `RUST_LOG`
  environment variable (e.g. `RUST_LOG=info`).
- Replace `tfrecord` by our own minimalist TensorBoard summary writing, removing
  92 dependencies.

### Removed

- Support for parsing as sequence labeling is removed in favor of biaffine
  parsing. Biaffine parsing results in higher accuracies and does not require
  tailored part-of-speech tags.
- Support for hard loss is removed from the distillation subcommand. Hard loss
  never worked well compared to soft loss.
