{ fetchurl }:

{
  # Pretrained models
  ALBERT_BASE_V2 = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2.pt";
    hash = "sha256-OIMX9lAmcHs8DNGG5bmHf2yD5ZIevr97AbBm3GEwoRU=";
  };

  BERT_BASE_GERMAN_CASED = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased.pt";
    hash = "sha256-lCjjTxpd8BATF50KA3tZbdzmXwQaBbg4MKKexRTevlY=";
  };

  SQUEEZEBERT_UNCASED = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/squeezebert-uncased.pt";
    hash = "sha256-dW1SYPmve5pPcBebkJP87lxzKex+cbu5y8loqulGn64=";
  };

  XLM_ROBERTA_BASE = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base.pt";
    hash = "sha256-PHU0/oswr8jJkZBySpOuPTDpRlbpiXvMMAAwmuqC8Ps=";
  };

  # Tokenization models
  ALBERT_BASE_V2_SENTENCEPIECE = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2-sentencepiece.model";
    hash = "sha256-/vsCtmemxcL+J2AtKOX7NCj2aricfW84jnyNRKAtAzY=";
  };

  BERT_BASE_GERMAN_CASED_VOCAB = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased-vocab.txt";
    hash = "sha256-cuEn+OcGxk9xOBNSbmi9E4BdFpHj8E1HCJEmTO3gIT0=";
  };

  XLM_ROBERTA_BASE_SENTENCEPIECE = fetchurl {
    url = "https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base-sentencepiece.bpe.model";
    hash = "sha256-z8gUar4qBIjp4qDFbeeVL3wRqwWeyhRaCnJ6/ODbKGU=";
  };
}
