# SyntaxDot

## Introduction

**SyntaxDot** is a sequence labeler using
[Transformer](https://arxiv.org/abs/1706.03762) networks. SyntaxDot
models can be trained from scratch or using pretrained models, such as
[BERT](https://arxiv.org/abs/1810.04805v2) or
[XLM-RoBERTa](https://arxiv.org/abs/1911.02116).

In principle, SyntaxDot can be used to perform any sequence labeling
task, but so far the focus has been on:

* Part-of-speech tagging
* Morphological tagging
* Topological field tagging
* Lemmatization
* Dependency parsing
* Named entity recognition

## Features

* Input representations:
  - Word pieces
  - Sentence pieces
* Flexible sequence encoder/decoder architecture, which supports:
  * Simple sequence labels (e.g. POS, morphology, named entities)
  * Lemmatization, based on edit trees
  * Dependency parsing
  * Simple API to extend to other tasks
* Models representations:
  * Transformers
  * Pretraining from BERT and XLM-RoBERTa models
* Multi-task training and classification using scalar weighting.
* Model distillation
* Deployment:
  * Standalone binary that links against PyTorch's `libtorch`
  * Very liberal [license](LICENSE.md)

## Documentation

* [Installation](doc/install.md)

## References

SyntaxDot uses techniques from or was inspired by the following papers:

* The model architecture and training regime was largely based on [75
  Languages, 1 Model: Parsing Universal Dependencies
  Universally](https://www.aclweb.org/anthology/D19-1279.pdf).  Dan
  Kondratyuk and Milan Straka, 2019, Proceedings of the EMNLP 2019 and
  the 9th IJCNLP.
* The encoding of lemmatization as edit trees was proposed in [Towards
  a Machine-Learning Architecture for Lexical Functional Grammar
  Parsing](http://grzegorz.chrupala.me/papers/phd-single.pdf).
  Grzegorz Chrupała, 2008, PhD dissertation, Dublin City University.

## Issues

You can report bugs and feature requests in the [SyntaxDot issue
tracker](https://github.com/tensordot/syntaxdot/issues).

## License

For licensing information, see [COPYRIGHT.md](COPYRIGHT.md).
