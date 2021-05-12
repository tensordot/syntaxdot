use serde::{Deserialize, Serialize};

use crate::models::traits::WordEmbeddingsConfig;

/// Bert model configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct BertConfig {
    pub attention_probs_dropout_prob: f64,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
}

impl Default for BertConfig {
    fn default() -> Self {
        BertConfig {
            attention_probs_dropout_prob: 0.1,
            hidden_act: "gelu".to_owned(),
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            layer_norm_eps: 1e-12,
            max_position_embeddings: 512,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            type_vocab_size: 2,
            vocab_size: 30000,
        }
    }
}

impl WordEmbeddingsConfig for BertConfig {
    fn dims(&self) -> i64 {
        self.hidden_size
    }

    fn dropout(&self) -> f64 {
        self.hidden_dropout_prob
    }

    fn initializer_range(&self) -> f64 {
        self.initializer_range
    }

    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }

    fn vocab_size(&self) -> i64 {
        self.vocab_size
    }
}
