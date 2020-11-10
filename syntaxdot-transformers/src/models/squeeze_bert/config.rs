use serde::{Deserialize, Serialize};

use crate::models::BertConfig;

/// SqueezeBert model configuration.
#[serde(default)]
#[derive(Debug, Deserialize, Serialize)]
pub struct SqueezeBertConfig {
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
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
    pub q_groups: i64,
    pub k_groups: i64,
    pub v_groups: i64,
    pub post_attention_groups: i64,
    pub intermediate_groups: i64,
    pub output_groups: i64,
}

impl Default for SqueezeBertConfig {
    fn default() -> Self {
        SqueezeBertConfig {
            attention_probs_dropout_prob: 0.1,
            embedding_size: 768,
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
            vocab_size: 30528,
            q_groups: 4,
            k_groups: 4,
            v_groups: 4,
            post_attention_groups: 1,
            intermediate_groups: 4,
            output_groups: 4,
        }
    }
}

impl From<&SqueezeBertConfig> for BertConfig {
    fn from(squeeze_bert_config: &SqueezeBertConfig) -> Self {
        BertConfig {
            attention_probs_dropout_prob: squeeze_bert_config.attention_probs_dropout_prob,
            hidden_act: squeeze_bert_config.hidden_act.clone(),
            hidden_dropout_prob: squeeze_bert_config.hidden_dropout_prob,
            hidden_size: squeeze_bert_config.hidden_size,
            initializer_range: squeeze_bert_config.initializer_range,
            intermediate_size: squeeze_bert_config.intermediate_size,
            layer_norm_eps: squeeze_bert_config.layer_norm_eps,
            max_position_embeddings: squeeze_bert_config.max_position_embeddings,
            num_attention_heads: squeeze_bert_config.num_attention_heads,
            num_hidden_layers: squeeze_bert_config.num_hidden_layers,
            type_vocab_size: squeeze_bert_config.type_vocab_size,
            vocab_size: squeeze_bert_config.vocab_size,
        }
    }
}
