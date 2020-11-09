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
  ["ALBERT_BASE_V2"]="https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2.hdf5"
  ["BERT_BASE_GERMAN_CASED"]="https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased.hdf5"
  ["SQUEEZEBERT_UNCASED"]="https://s3.tensordot.com/syntaxdot/pretrained/squeezebert-uncased.hdf5"
  ["XLM_ROBERTA_BASE"]="https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base.hdf5"

  ["ALBERT_BASE_V2_SENTENCEPIECE"]="https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2-sentencepiece.model"
  ["BERT_BASE_GERMAN_CASED_VOCAB"]="https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased-vocab.txt"
  ["XLM_ROBERTA_BASE_SENTENCEPIECE"]="https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base-sentencepiece.bpe.model"
)

if [ ! -d "$cache_dir" ]; then
  mkdir -p "$cache_dir"
fi

for var in "${!models[@]}"; do
  url="${models[$var]}"
  data="${cache_dir}/$(basename "${url}")"

  if [ ! -e "${data}" ]; then
    curl -o "${data}" "${url}"
  fi

  declare -x "${var}"="${data}"
done

( cd syntaxdot-transformers ; cargo test --features model-tests )

( cd syntaxdot ; cargo test --features model-tests )