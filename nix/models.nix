let
  sources = import ./sources.nix;
in {
  # Pretrained models for testing.
  ALBERT_BASE_V2 = sources.albert-base-v2;
  BERT_BASE_GERMAN_CASED = sources.bert-base-german-cased;
  XLM_ROBERTA_BASE = sources.xlm-roberta-base;

  # Vocabularies for testing.
  ALBERT_BASE_V2_SENTENCEPIECE = sources.albert-base-v2-sentencepiece;
  BERT_BASE_GERMAN_CASED_VOCAB = sources.bert-base-german-cased-vocab;
  XLM_ROBERTA_BASE_SENTENCEPIECE = sources.xlm-roberta-base-sentencepiece;
}
