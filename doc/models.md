# Models

The following ready-to-use models are available. For each language,
treebanks were shuffled and split 7/1/2 in train/development/held-out
partitions. Reported accuracies are from evaluation on held-out data.

Performance in sentences per second is measured using:

* CPU: Ryzen 3700X CPU, using 4 threads.
* GPU: NVIDIA RTX 2060

Models can be used by unpacking the model archive and running

```bash
$ syntaxdot annotate model-name/syntaxdot.conf
```

## Dutch

| Model                                                                                                     | UD POS | Lemma | UD morph |   LAS | Size (MiB) | CPU sent/s | GPU sents/s |
|:----------------------------------------------------------------------------------------------------------|-------:|------:|---------:|------:|-----------:|-----------:|------------:|
| [Finetuned XLM-R base](https://s3.tensordot.com/syntaxdot/models/nl-ud-huge-20210301.tar.gz)              |  98.90 | 99.03 |    98.87 | 94.37 |       1087 |         61 |         755 |
| [Distilled, 12 layers, 384 hidden](https://s3.tensordot.com/syntaxdot/models/nl-ud-large-20210324.tar.gz) |  98.83 | 99.03 |    98.80 | 93.91 |        200 |        135 |        1450 |
| [Distilled, 6 layers, 384 hidden](https://s3.tensordot.com/syntaxdot/models/nl-ud-medium-20210312.tar.gz) |  98.80 | 99.05 |    98.79 | 93.42 |        133 |        240 |        2359 |

## German

| Model                                                                                                     | UD POS | STTS POS | Lemma | UD morph | TDZ morph |   LAS | Topo Field | Size (MiB) | CPU sent/s | GPU sent/s |
|:----------------------------------------------------------------------------------------------------------|-------:|---------:|------:|---------:|----------:|------:|-----------:|-----------:|-----------:|-----------:|
| [Finetuned XLM-R base](https://github.com/tensordot/syntaxdot-models/releases/download/de-ud-2021/de-ud-huge-20210307.tar.gz)              |  99.54 |    99.48 | 99.34 |    98.38 |     98.43 | 96.59 |      98.17 |       1087 |         45 |        614 |
| [Distilled, 12 layers, 384 hidden](https://github.com/tensordot/syntaxdot-models/releases/download/de-ud-2021/de-ud-large-20210326.tar.gz) |  99.50 |    99.44 | 99.31 |    98.31 |     98.36 | 96.17 |      98.12 |        208 |        105 |       1131 |
| [Distilled, 6 layers, 384 hidden](https://github.com/tensordot/syntaxdot-models/releases/download/de-ud-2021/de-ud-medium-20210326.tar.gz)  |  99.46 |    99.40 | 99.29 |    98.20 |     98.27 | 95.48 |      97.97 |        140 |        180 |       1748 |

