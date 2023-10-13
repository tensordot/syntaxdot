#!/usr/bin/env bash

# This script runs all tests, including model tests. In order to do
# so, models are downloaded and stored in $XDG_CACHE_HOME/syntaxdot.

set -euo pipefail
IFS=$'\n\t'

if ! [ -x "$(command -v curl)" ] ; then
  >&2 echo "'curl' is required for downloading test data"
  exit 1
fi

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/syntaxdot"

declare -A models=(
  ["ALBERT_BASE_V2"]="https://f001.backblazeb2.com/file/danieldk-blob/syntaxdot/albert-base-v2.pt"
  ["BERT_BASE_GERMAN_CASED"]="https://f001.backblazeb2.com/file/danieldk-blob/syntaxdot/bert-base-german-cased.pt"
  ["SQUEEZEBERT_UNCASED"]="https://f001.backblazeb2.com/file/danieldk-blob/syntaxdot/squeezebert-base-uncased.pt"
  ["XLM_ROBERTA_BASE"]="https://f001.backblazeb2.com/file/danieldk-blob/syntaxdot/xlm-roberta-base.pt"

  ["ALBERT_BASE_V2_SENTENCEPIECE"]="https://huggingface.co/albert-base-v2/resolve/main/spiece.model"
  ["BERT_BASE_GERMAN_CASED_VOCAB"]="https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt"
  ["XLM_ROBERTA_BASE_SENTENCEPIECE"]="https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"
)

if [ ! -d "$cache_dir" ]; then
  mkdir -p "$cache_dir"
fi

for var in "${!models[@]}"; do
  url="${models[$var]}"
  data="${cache_dir}/$(basename "${url}")"

  # Since these checkpoints were generated, an assumption was added that
  # .pt files are created from Python code. Rename to .ot to avoid loading
  # issues.
  data=${data/%.pt/.ot}

  if [ ! -e "${data}" ]; then
    curl -fo "${data}" "${url}"
  fi

  declare -x "${var}"="${data}"
done

# Regular tests for all crates
cargo test

# Regular tests + model tests for transformers and tokenizers
( cd syntaxdot-tokenizers ; cargo test --features model-tests )
( cd syntaxdot-transformers ; cargo test --features model-tests )
