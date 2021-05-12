use serde::{Deserialize, Serialize};

use crate::models::bert::BertConfig;
use crate::models::traits::WordEmbeddingsConfig;

/// ALBERT model configuration.
#[derive(Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AlbertConfig {
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        AlbertConfig {
            attention_probs_dropout_prob: 0.,
            embedding_size: 128,
            hidden_act: "gelu_new".to_owned(),
            hidden_dropout_prob: 0.,
            hidden_size: 768,
            initializer_range: 0.02,
            inner_group_num: 1,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            num_hidden_groups: 1,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30000,
        }
    }
}

impl From<&AlbertConfig> for BertConfig {
    fn from(albert_config: &AlbertConfig) -> Self {
        BertConfig {
            attention_probs_dropout_prob: albert_config.attention_probs_dropout_prob,
            hidden_act: albert_config.hidden_act.clone(),
            hidden_dropout_prob: albert_config.hidden_dropout_prob,
            hidden_size: albert_config.hidden_size,
            initializer_range: albert_config.initializer_range,
            intermediate_size: albert_config.intermediate_size,
            layer_norm_eps: 1e-12,
            max_position_embeddings: albert_config.max_position_embeddings,
            num_attention_heads: albert_config.num_attention_heads,
            num_hidden_layers: albert_config.num_hidden_layers,
            type_vocab_size: albert_config.type_vocab_size,
            vocab_size: albert_config.vocab_size,
        }
    }
}

impl WordEmbeddingsConfig for AlbertConfig {
    fn dims(&self) -> i64 {
        self.embedding_size
    }

    fn dropout(&self) -> f64 {
        self.hidden_dropout_prob
    }

    fn initializer_range(&self) -> f64 {
        self.initializer_range
    }

    fn layer_norm_eps(&self) -> f64 {
        1e-12
    }

    fn vocab_size(&self) -> i64 {
        self.vocab_size
    }
}
