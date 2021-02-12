# Finetuning

## Introduction

A SyntaxDot model is trained using *finetuning*. Finetuning takes a
pretrained (transformer-based) language model and refines its weights
for syntax annotation.

To finetune a model, you need several ingredients:

* A pretrained language model, including its vocabulary and model
  configuration.
* Training and test data. The training data will be used to finetune
  the model's parameters. The test data is used to evaluate the
  model's performance during finetuning.
* A SyntaxDot configuration file.

## Getting a pretrained model

SyntaxDot requires pretrained models in the libtorch OutputArchive
format, with the expected parameter names. Python scripts to convert
some commonly-used model types to the format expected by SyntaxDot are
provided in the [../scripts](scripts) directory.

You could also use an existing pretrained model in the SyntaxDot
format. You can download the required parts for the well-performing
multilingual
[XLM-RoBERTa](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/)
base model here:

* [Model parameters](https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base.pt)
* [Model configuration](https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base.json)
* [Model vocabulary](https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base-sentencepiece.bpe.model)

## SyntaxDot configuration file

The SyntaxDot configuration file contains the SyntaxDot-specific model
settings in TOML format, such as the sequence labeling tasks, and
biaffine parsing.  This is an example configuration file:

```toml
[input]
tokenizer = { xlm_roberta = { vocab = "xlm-roberta-base-sentencepiece.bpe.model" } }

[biaffine]
labels = "syntaxdot.biaffine"
head = { dims = 128, head_bias = true, dependent_bias = true }
relation = { dims = 128, head_bias = true, dependent_bias = true }

[labeler]
labels = "sticker.labels"
encoders = [
  { name = "pos-ud", encoder = { sequence = "upos" } },
  { name = "pos-extended", encoder = { sequence = "xpos" } },
  { name = "lemma", encoder = { lemma = "form" } },
  { name = "morph", encoder = { sequence = "feature_string" } },
]

[model]
pretrain_type = "xlm_roberta"
parameters = "epoch-74"
position_embeddings = "model"
pretrain_config = "xlm-roberta-base-config.json"
```

We will now discuss each section in more detail.

### `input`

The `input` section describes processing of the input tokens. Currently,
this section only configures the tokenizer that should be used. For
example,

```toml
[input]
tokenizer = { xlm_roberta = { vocab = "xlm-roberta-base-sentencepiece.bpe.model" } }
```

Uses the XLM-RoBERTa tokenization model, which is in the file
`xlm-roberta-base-sentencepiece.bpe.model`. The supported tokenizers
are:

* `albert`: for ALBERT models.
* `bert`: for BERT-based models
* `xlm_roberta`: for XLM-RoBERTa models.

### `biaffine`

The `biaffine` section configures a biaffine parser. If this section
is absent, the model is a pure sequence labeling model, and cannot
perform dependency parsing. Here is an example `biaffine` section:

```
[biaffine]
labels = "syntaxdot.biaffine"
head = { dims = 128, head_bias = true, dependent_bias = true }
relation = { dims = 128, head_bias = true, dependent_bias = true }
```

The `labels` options sets the file name to which the dependency labels
should be saved. This is only done once during model preparation (see
below).

The `head` and `relation` options configure the hidden representations
of tokens as used for head prediction and relation prediction. In both
cases, `dims` configures the dimensionality of the hidden
representations. `head_bias` and `dependent_bias` configure whether a
bias should be added for a token's representation as a head or as a
dependent. Enabling these biases generally doesn't negatively affect
performance, so it is safest to simply enable them.

### `labeler`

This section configures one or more sequence labelers. For example:

```
[labeler]
labels = "sticker.labels"
encoders = [
  { name = "pos-ud", encoder = { sequence = "upos" } },
  { name = "pos-extended", encoder = { sequence = "xpos" } },
  { name = "lemma", encoder = { lemma = "form" } },
  { name = "morph", encoder = { sequence = "feature_string" } },
]
```

The `labels` option specifies the file in which all labels for
sequence annotation should be stored. `encoders` is a list that
configures the different sequence annotation layers. Each layer should
have a unique `name` and specify an encoder. We will now discuss
the encoder types in more detail.

#### `sequence`

The sequence encoder is the simplest form of encoder, it simply takes
values from a CoNLL-U layer. Basic sequence encoders have the
following form:

```
{ name = "NAME", encoder = { sequence = "LAYER" } },
```

where LAYER is one of:

* `upos`: universal part-of-speech tag
* `xpos`: language-specific part-of-speech tag
* `feature_string`: the features field as a string

Note that `feature_string` is invariant to the feature ordering, since
features are internally sorted in lexicographical order.

It is also possible to create annotation layers based on specific
features or miscellaneous features. For example, the following
sequence encoder uses the `TopoField` feature of the miscellaneous
features, setting the default value to `null` if the feature
is absent:

```
{ name = "tf", encoder = { sequence = { misc = { feature = "TopoField", default = "null" } } } }
```

Similarly, we could create an annotation layer using the `Number`
feature:

```
{ name = "tf", encoder = { sequence = { misc = { feature = "TopoField", default = "null" } } } }
```

#### `lemma`

The lemma encoder (transparently) encodes lemmas to edit trees. The
configuration of this encoder is simply as follows:

```
{ name = "lemma", encoder = { lemma = "form" } },
```

The only configuration is the back-off strategy, which is used when
the predicted edit tree cannot be applied. Here the back-off strategy
is to use the form as the lemma. Another possible value is `nothing`,
which will not update the lemma at all in such a case.

### `model`

The final configuration section of the SyntaxDot configuration
specifies the model. For example:

```toml
[model]
pretrain_type = "xlm_roberta"
parameters = "epoch-74"
position_embeddings = "model"
pretrain_config = "xlm-roberta-base-config.json"
```

`pretrain_type` is the type of pretrained model that is used. This is
one of:

* `albert`
* `bert`
* `squeeze_albert`
* `squeeze_bert`
* `xlm_roberta`

The `parameters` option contains the file with model parameters. This
can be any value during finetuning. But after finetuning, this should
be set to the parameter file of the best epoch.

The `position_embeddings` option configures the type of position
embeddings that should be used. For finetuning existing models, this
should always be set to `model`, which will use the position
embeddings of the model.

Finally, `pretrain_config` is the filename of the pretrained model
configuration.

## Finetuning

With the configuration set up, the a model can be trained. The first
step is to prepare the label files, which is done with the `syntaxdot
prepare` subcommand:

```shell
$ syntaxdot prepare syntaxdot.conf train.conllu
```

Then, you can start the finetuning of the pretrained model. For
example, to finetune a syntax model with XLM-RoBERTa base:

```shell
$ syntaxdot finetune \
	syntaxdot.conf \
	xlm-roberta-base.pt \
	train.conllu \
	test.conllu \
	--gpu 0 \
	--label-smoothing 0.03 \
	--maxlen 100 \
	--mixed-precision \
	--warmup 10000
```

SyntaxDot has more command-line options to configure finetuning, but is
the recommended set of options:

* `--gpu 0`: enables GPU training using GPU 0.
* `--label-smoothin 0.03`: enables label smoothing, distributing 0.03
  of the 'probability mass' to incorrect labels. This is a form of
  regularization and generally results in better models.
* `--maxlen 100`: excludes sentences of more than 100 pieces, this can
  be used in conjunction with the `batch-size` option to control the
  amount of GPU memory needed during training.
* `--mixed-precision`: use mixed precision. This lowers GPU memory use
  and speeds up training considerably, but is only supported on NVIDIA
  GPUs with Tensor Cores (Volta/Turing GPUs or later generations).
* `--warmup 10000`: use 10,000 steps of learning rate warmup. This
  avoids that early weight update steps are too large.

After finetuning is done, SyntaxDot will report the best epoch. Don't
forget to update the `parameters` option in your SyntaxDot
configuration to use the parameters from the best epoch!
