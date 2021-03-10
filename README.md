# SyntaxDot

## Introduction

**SyntaxDot** is a sequence labeler and dependency parser using
[Transformer](https://arxiv.org/abs/1706.03762) networks. SyntaxDot models can
be trained from scratch or using pretrained models, such as
[BERT](https://arxiv.org/abs/1810.04805v2) or
[XLM-RoBERTa](https://arxiv.org/abs/1911.02116).

In principle, SyntaxDot can be used to perform any sequence labeling
task, but so far the focus has been on:

* Part-of-speech tagging
* Morphological tagging
* Topological field tagging
* Lemmatization
* Named entity recognition

The easiest way to get started with SyntaxDot is to [use a pretrained
sticker2
model](https://github.com/stickeritis/sticker2/blob/master/doc/pretrained.md)
(SyntaxDot is currently compatbile with sticker2 models).

## Features

* Input representations:
  - Word pieces
  - Sentence pieces
* Flexible sequence encoder/decoder architecture, which supports:
  * Simple sequence labels (e.g. POS, morphology, named entities)
  * Lemmatization, based on edit trees
  * Simple API to extend to other tasks
* Dependency parsing using deep biaffine attention and MST decoding.
* Multi-task training and classification using scalar weighting.
* Encoder models:
  * Transformers
  * Finetuning of BERT, XLM-RoBERTa, ALBERT, and SqueezeBERT models
* Model distillation
* Deployment:
  * Standalone binary that links against PyTorch's `libtorch`
  * Very liberal [license](LICENSE.md)

## Documentation

* [Installation](doc/install.md)
* [Finetuning](doc/finetune.md) (training)

## References

SyntaxDot uses techniques from or was inspired by the following papers:

* The biaffine dependency parsing layer is based on [Deep biaffine attention for
  neural dependency parsing](https://arxiv.org/pdf/1611.01734.pdf).
  Timothy Dozat and Christopher Manning, ICLR 2017.
* The model architecture and training regime was largely based on [75
  Languages, 1 Model: Parsing Universal Dependencies
  Universally](https://www.aclweb.org/anthology/D19-1279.pdf).  Dan
  Kondratyuk and Milan Straka, 2019, Proceedings of the EMNLP 2019 and
  the 9th IJCNLP.
* The tagging as sequence labeling scheme was proposed by [Dependency
  Parsing as a Sequence Labeling
  Task](https://www.degruyter.com/downloadpdf/j/pralin.2010.94.issue--1/v10108-010-0017-3/v10108-010-0017-3.pdf). Drahomíra
  Spoustová, Miroslav Spousta, 2010, The Prague Bulletin of
  Mathematical Linguistics, Volume 94.
* The idea to combine this scheme with neural networks comes from
  [Viable Dependency Parsing as Sequence
  Labeling](https://www.aclweb.org/anthology/papers/N/N19/N19-1077/). Michalina
  Strzyz, David Vilares, Carlos Gómez-Rodríguez, 2019, Proceedings of
  the 2019 Conference of the North American Chapter of the Association
  for Computational Linguistics: Human Language Technologies
* The encoding of lemmatization as edit trees was proposed in [Towards
  a Machine-Learning Architecture for Lexical Functional Grammar
  Parsing](http://grzegorz.chrupala.me/papers/phd-single.pdf).
  Grzegorz Chrupała, 2008, PhD dissertation, Dublin City University.

## Issues

You can report bugs and feature requests in the [SyntaxDot issue
tracker](https://github.com/tensordot/syntaxdot/issues).

## License

For licensing information, see [COPYRIGHT.md](COPYRIGHT.md).
