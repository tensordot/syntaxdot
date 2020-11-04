//! SqueezeBERT (Iandola et al., 2020) + ALBERT (Lan et al., 2020)
//!
//! This model combines SqueezeBERT and ALBERT:
//!
//! * SqueezeBERT replaces most matrix multiplications by grouped
//!   convolutions, resulting in smaller models and higher inference
//!   speeds.
//! * ALBERT allows layers to share parameters and decouples the
//!   embedding size from the hidden state size. Both result in
//!   smaller models.
//!
//! Combined, your models can be even smaller and faster.

use std::borrow::Borrow;

use serde::{Deserialize, Serialize};
use tch::nn::Module;
use tch::Tensor;
use tch_ext::PathExt;

use crate::models::albert::{AlbertConfig, AlbertEmbeddingProjection};
use crate::models::bert::{BertConfig, BertError};
use crate::models::layer_output::LayerOutput;
use crate::models::squeeze_bert::{SqueezeBertConfig, SqueezeBertLayer};
use crate::models::traits::WordEmbeddingsConfig;
use crate::models::Encoder;
use crate::util::LogitsMask;

/// SqueezeALBERT model configuration.
///
/// This is the union set of the SqueezeBERT and ALBERT configurations:
///
/// * SqueezeBERT uses `q_groups`, `k_groups`, `v_groups`,
///   `post_attention_groups`, `intermediate_groups`, and
///   `output_groups` to configure the number of groups in grouped
///   convolutions.
/// * ALBERT uses `num_hidden_groups` to configure the number of layer
///   groups and `embedding_size` to configure the size of piece
///   embeddings.
#[serde(default)]
#[derive(Debug, Deserialize, Serialize)]
pub struct SqueezeAlbertConfig {
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
    pub q_groups: i64,
    pub k_groups: i64,
    pub v_groups: i64,
    pub post_attention_groups: i64,
    pub intermediate_groups: i64,
    pub output_groups: i64,
}

impl Default for SqueezeAlbertConfig {
    fn default() -> Self {
        SqueezeAlbertConfig {
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
            q_groups: 4,
            k_groups: 4,
            v_groups: 4,
            post_attention_groups: 1,
            intermediate_groups: 4,
            output_groups: 4,
        }
    }
}

impl From<&SqueezeAlbertConfig> for AlbertConfig {
    fn from(albert_config: &SqueezeAlbertConfig) -> Self {
        AlbertConfig {
            attention_probs_dropout_prob: albert_config.attention_probs_dropout_prob,
            embedding_size: albert_config.embedding_size,
            hidden_act: albert_config.hidden_act.clone(),
            hidden_dropout_prob: albert_config.hidden_dropout_prob,
            hidden_size: albert_config.hidden_size,
            initializer_range: albert_config.initializer_range,
            inner_group_num: albert_config.inner_group_num,
            intermediate_size: albert_config.intermediate_size,
            max_position_embeddings: albert_config.max_position_embeddings,
            num_attention_heads: albert_config.num_attention_heads,
            num_hidden_groups: albert_config.num_hidden_groups,
            num_hidden_layers: albert_config.num_hidden_layers,
            type_vocab_size: albert_config.type_vocab_size,
            vocab_size: albert_config.vocab_size,
        }
    }
}

impl From<&SqueezeAlbertConfig> for BertConfig {
    fn from(albert_config: &SqueezeAlbertConfig) -> Self {
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

impl From<&SqueezeAlbertConfig> for SqueezeBertConfig {
    fn from(config: &SqueezeAlbertConfig) -> Self {
        SqueezeBertConfig {
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            embedding_size: config.embedding_size,
            hidden_act: config.hidden_act.clone(),
            hidden_dropout_prob: config.hidden_dropout_prob,
            hidden_size: config.hidden_size,
            initializer_range: config.initializer_range,
            intermediate_size: config.intermediate_size,
            layer_norm_eps: config.layer_norm_eps(),
            max_position_embeddings: config.max_position_embeddings,
            num_attention_heads: config.num_attention_heads,
            num_hidden_layers: config.num_hidden_layers,
            type_vocab_size: config.type_vocab_size,
            vocab_size: config.vocab_size,
            q_groups: config.q_groups,
            k_groups: config.k_groups,
            v_groups: config.v_groups,
            post_attention_groups: config.post_attention_groups,
            intermediate_groups: config.intermediate_groups,
            output_groups: config.output_groups,
        }
    }
}

impl WordEmbeddingsConfig for SqueezeAlbertConfig {
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

/// SqueezeALBERT encoder.
///
/// This encoder uses the SqueezeBERT encoder with two modifications:
///
/// 1. The embeddings are projected to fit the hidden layer size.
/// 2. Weights are shared between layers.
#[derive(Debug)]
pub struct SqueezeAlbertEncoder {
    groups: Vec<SqueezeBertLayer>,
    n_layers: i64,
    projection: AlbertEmbeddingProjection,
}

impl SqueezeAlbertEncoder {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &SqueezeAlbertConfig,
    ) -> Result<Self, BertError> {
        assert!(
            config.num_hidden_groups > 0,
            "Need at least 1 hidden group, got: {}",
            config.num_hidden_groups
        );

        let vs = vs.borrow();

        let mut groups = Vec::with_capacity(config.num_hidden_groups as usize);
        for group_idx in 0..config.num_hidden_groups {
            groups.push(SqueezeBertLayer::new(
                vs.sub(format!("group_{}", group_idx)).sub("inner_group_0"),
                &config.into(),
            )?);
        }
        let albert_config: AlbertConfig = config.into();
        let projection = AlbertEmbeddingProjection::new(vs, &albert_config);

        Ok(SqueezeAlbertEncoder {
            groups,
            n_layers: config.num_hidden_layers,
            projection,
        })
    }
}

impl Encoder for SqueezeAlbertEncoder {
    fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Vec<LayerOutput> {
        let hidden_states = self.projection.forward(&input);

        let input = hidden_states.permute(&[0, 2, 1]);

        let mut all_layer_outputs = Vec::with_capacity(self.n_layers as usize + 1);
        all_layer_outputs.push(LayerOutput::Embedding(hidden_states.shallow_clone()));

        let attention_mask = attention_mask.map(|mask| LogitsMask::from_bool_mask(mask));

        let layers_per_group = self.n_layers as usize / self.groups.len();

        let mut hidden_states = input;
        for idx in 0..self.n_layers {
            let layer_output = self.groups[idx as usize / layers_per_group].forward_t(
                &hidden_states,
                attention_mask.as_ref(),
                train,
            );

            hidden_states = layer_output.output().shallow_clone();

            all_layer_outputs.push(layer_output);
        }

        // Convert hidden states to [batch_size, seq_len, hidden_size].
        for layer_output in &mut all_layer_outputs {
            *layer_output.output_mut() = layer_output.output().permute(&[0, 2, 1]);
        }

        all_layer_outputs
    }

    fn n_layers(&self) -> i64 {
        self.n_layers + 1
    }
}
