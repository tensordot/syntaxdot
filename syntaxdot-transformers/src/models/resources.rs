use std::env;

use cached_path::Cache;
use hdf5::File;
use lazy_static::lazy_static;

const ALBERT_BASE_V2_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/albert-base-v2.hdf5";

const BERT_BASE_GERMAN_CASED_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/bert-base-german-cased.hdf5";

const SQUEEZEBERT_UNCASED_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/squeezebert-uncased.hdf5";

const XLM_ROBERTA_BASE_URL: &str =
    "https://s3.tensordot.com/syntaxdot/pretrained/xlm-roberta-base.hdf5";

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
    pub static ref ALBERT_BASE_V2: File = {
        let path = RESOURCE_CACHE.cached_path(ALBERT_BASE_V2_URL).unwrap();
        File::open(path).unwrap()
    };
    pub static ref BERT_BASE_GERMAN: File = {
        let path = RESOURCE_CACHE
            .cached_path(BERT_BASE_GERMAN_CASED_URL)
            .unwrap();
        File::open(path).unwrap()
    };
    pub static ref SQUEEZEBERT_UNCASED: File = {
        let path = RESOURCE_CACHE.cached_path(SQUEEZEBERT_UNCASED_URL).unwrap();
        File::open(path).unwrap()
    };
    pub static ref XLM_ROBERTA_BASE: File = {
        let path = RESOURCE_CACHE.cached_path(XLM_ROBERTA_BASE_URL).unwrap();
        File::open(path).unwrap()
    };
}
