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

| Model                                                                                     | UD POS | Lemma | UD morph |   LAS | Size (MiB) | CPU sent/s | GPU sents/s |
|:------------------------------------------------------------------------------------------|-------:|------:|---------:|------:|-----------:|-----------:|------------:|
| [Finetuned XLM-R base](https://s3.tensordot.com/syntaxdot/models/nl-xlmr-20210301.tar.gz) |  98.90 | 99.03 |    98.87 | 94.37 |       1087 |         61 |         755 |

## German

| Model                                                                                     | UD POS | STTS POS | Lemma | UD morph | TDZ morph |   LAS | Topo Field | Size (MiB) | CPU sent/s | GPU sent/s |
|:------------------------------------------------------------------------------------------|-------:|---------:|------:|---------:|----------:|------:|-----------:|-----------:|-----------:|-----------:|
| [Finetuned XLM-R base](https://s3.tensordot.com/syntaxdot/models/de-xlmr-20210307.tar.gz) |  99.54 |    99.48 | 99.34 |    98.38 |     98.43 | 96.59 |      98.17 |       1087 |         45 |        614 |

