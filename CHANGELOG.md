# Changelog

## [Unreleased]

### Added

- Add support for parallelizing annotation at the batch level. SyntaxDot has
  so far used PyTorch inter/intraop parallelization. This change adds
  support for parallelization at the batch level. Annotation-level
  parallelization can be configured with the `annotation-threads`
  command-line option of `syntaxdot annotate`.

### Changed

- SyntaxDot now uses dynamic batch sizes. Before this change, the batch
  size (`--batch-size`) was specified as the number of sentences per
  batch. Since sentences are sorted by length before batching, annotation
  is performed on batches with roughly equisized sequences. However,
  later batches required more computations per batch due to longer
  sequence lengths.

  This change replaces the `--batch-size` option by the `--max-batch-pieces`
  option. This option specifies the number of word/sentence pieces that
  a batch should contain. SyntaxDot annotation creates batches that contains
  at most that number of pieces. The only exception are single sentences
  that are longer than the maximum number of batch pieces.

  With this change, annotating each batch is approximately the same amount
  of work. This leads to approximately 10% increase in performance.

  Since the batch size is not fixed anymore, the readahead (`--readahead`)
  is now specified in number of sentences.
- Update to [libtorch
  1.9.0](https://github.com/pytorch/pytorch/releases/tag/v1.9.0) and
  [tch 0.5.0](https://github.com/LaurentMazare/tch-rs).
- Change the default number of inter/intraop threads to 1. Use 4 threads for
  annotation-level parallelization. This has shown to be faster for all models,
  both on AMD Ryzen and Apple M1.

## 0.3.1

### Fixed

- Apply biaffine dependency encoding before sequence labeling, so that
  the TüBa-D/Z lemma decoder has access to dependency relations.

## 0.3.0

### Added

- Support for biaffine dependency parsing (Dozat & Manning, 2016).
  Biaffine parsing is enabled through the `biaffine` configuration
  option.
- Support for pooling the pieces of a token by taking the mean of the
  pieces. This type of pooling is enabled by setting the
  `model.pooler` option to `mean`. The old behavior of discarding
  continuation pieces is used when this option is set to `discard`.
- Add the `keep-best` option to the `finetune` and `distill`
  subcommands. With this option only the parameter files for the N
  best epochs/steps are retained during distillation.
- Support for hidden layer distillation loss. This loss uses the mean
  squared error of the teacher's hidden layer representations and
  student representations for faster convergence.

### Changed

- Update to [libtorch
  1.8.0](https://github.com/pytorch/pytorch/releases/tag/v1.8.0) and
  [tch 0.4.0](https://github.com/LaurentMazare/tch-rs).
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
- Replace calls to the Rust Torch crate (`tch`) by fallible
  counterparts, this makes exceptions thrown by Torch far easier to
  read.
- Uses of the `eprintln!` macro are replaced by logging using `log` and
  `env_logger`. The verbosity of the logs can be controlled with the `RUST_LOG`
  environment variable (e.g. `RUST_LOG=info`).
- Replace `tfrecord` by our own minimalist TensorBoard summary writing, removing
  92 dependencies.

### Removed

- Support for hard loss is removed from the distillation subcommand. Hard loss
  never worked well compared to soft loss.

### Fixed

- Fix an off-by-one slicing error in `SequenceClassifiers::top_k`.
