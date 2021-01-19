# Changelog

## [Unreleased]

### Changed

- Implementations of `Tokenizer` are now required to put a piece that marks the
  beginning of a sentence before the first token piece. `BertTokenizer` was the
  only tokenizer that did not fulfill this requirement. `BertTokenizer` is
  updated to insert the `[CLS]` piece as a beginning of sentence marker.
  **Warning:** this breaks existing models with `tokenizer = "bert"`, which should
  be retrained.
