use std::env;
use std::path::PathBuf;

use cached_path::Cache;

use lazy_static::lazy_static;

const ALBERT_V2_SENTENCEPIECE_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2-sentencepiece.model";

const BERT_BASE_GERMAN_CASED_VOCAB_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased-vocab.txt";

const XLM_ROBERTA_BASE_SENTENCEPIECE_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base-sentencepiece.bpe.model";

lazy_static! {
    static ref RESOURCE_CACHE: Cache = {
        let cache_dir = match env::var("SYNTAXDOT_CACHE") {
            Ok(dir) => dir.into(),
            Err(_) => {
                let mut cache_dir = dirs::cache_dir().unwrap();
                cache_dir.push("syntaxdot");
                cache_dir
            }
        };

        Cache::builder()
            .dir(cache_dir)
            .freshness_lifetime(24 * 3600)
            .build()
            .unwrap()
    };
    pub static ref ALBERT_V2_SENTENCEPIECE: PathBuf = {
        RESOURCE_CACHE
            .cached_path(ALBERT_V2_SENTENCEPIECE_URL)
            .unwrap()
    };
    pub static ref BERT_BASE_GERMAN_CASED_VOCAB: PathBuf = {
        RESOURCE_CACHE
            .cached_path(BERT_BASE_GERMAN_CASED_VOCAB_URL)
            .unwrap()
    };
    pub static ref XLM_ROBERTA_BASE_SENTENCEPIECE: PathBuf = {
        RESOURCE_CACHE
            .cached_path(XLM_ROBERTA_BASE_SENTENCEPIECE_URL)
            .unwrap()
    };
}
