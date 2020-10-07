// Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2019 The sticker developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! BERT (Devlin et al., 2018)

use std::borrow::Borrow;
use std::iter;

use serde::{Deserialize, Serialize};
use tch::nn::{Init, Linear, Module, ModuleT, Path};
use tch::{Kind, Tensor};
use thiserror::Error;

use crate::activations;
use crate::cow::CowTensor;
use crate::layers::{Dropout, Embedding, LayerNorm};
use crate::models::encoder::Encoder;
use crate::models::traits::WordEmbeddingsConfig;
use crate::traits::{LayerAttention, LayerOutput};
use crate::util::LogitsMask;

/// Bert attention block.
#[derive(Debug)]
pub struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        BertAttention {
            self_attention: BertSelfAttention::new(vs / "self", config),
            self_output: BertSelfOutput::new(vs / "output", config),
        }
    }

    /// Apply the attention block.
    ///
    /// Outputs the hidden states and the attention probabilities.
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let (self_outputs, attention_probs) =
            self.self_attention
                .forward_t(hidden_states, attention_mask, train);
        let attention_output = self
            .self_output
            .forward_t(&self_outputs, &hidden_states, train);

        (attention_output, attention_probs)
    }
}

/// Bert model configuration.
#[serde(default)]
#[derive(Clone, Debug, Deserialize, Serialize)]
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

/// Construct the embeddings from word, position and token_type embeddings.
#[derive(Debug)]
pub struct BertEmbeddings {
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    word_embeddings: Embedding,

    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    /// Construct new Bert embeddings with the given variable store
    /// and Bert configuration.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        let normal_init = Init::Randn {
            mean: 0.,
            stdev: config.initializer_range,
        };

        let word_embeddings = Embedding::new(
            vs / "word_embeddings",
            "embeddings",
            config.vocab_size,
            config.hidden_size,
            normal_init,
        );

        let position_embeddings = Embedding::new(
            vs / "position_embeddings",
            "embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            normal_init,
        );

        let token_type_embeddings = Embedding::new(
            vs / "token_type_embeddings",
            "embeddings",
            config.type_vocab_size,
            config.hidden_size,
            normal_init,
        );

        let layer_norm = LayerNorm::new(
            vs / "layer_norm",
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );

        let dropout = Dropout::new(config.hidden_dropout_prob);

        BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let input_shape = input_ids.size();

        let seq_length = input_shape[1];
        let device = input_ids.device();

        let position_ids = match position_ids {
            Some(position_ids) => CowTensor::Borrowed(position_ids),
            None => CowTensor::Owned(
                Tensor::arange(seq_length, (Kind::Int64, device))
                    .unsqueeze(0)
                    // XXX: Second argument is 'implicit', do we need to set this?
                    .expand(&input_shape, false),
            ),
        };

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => CowTensor::Borrowed(token_type_ids),
            None => CowTensor::Owned(Tensor::zeros(&input_shape, (Kind::Int64, device))),
        };

        let input_embeddings = self.word_embeddings.forward(input_ids);
        let position_embeddings = self.position_embeddings.forward(&*position_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(&*token_type_ids);

        let embeddings = input_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(&embeddings);
        self.dropout.forward_t(&embeddings, train)
    }
}

impl ModuleT for BertEmbeddings {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.forward(input, None, None, train)
    }
}

#[derive(Debug)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let layers = (0..config.num_hidden_layers)
            .map(|layer| BertLayer::new(vs / format!("layer_{}", layer), config))
            .collect::<Result<_, _>>()?;

        Ok(BertEncoder { layers })
    }
}

impl Encoder for BertEncoder {
    fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Vec<BertLayerOutput> {
        let mut all_layer_outputs = Vec::with_capacity(self.layers.len());

        let attention_mask = attention_mask.map(|mask| LogitsMask::from_bool_mask(mask));

        let mut hidden_states = CowTensor::Borrowed(input);
        for layer in &self.layers {
            let layer_output = layer.forward_t(&hidden_states, attention_mask.as_ref(), train);

            hidden_states = CowTensor::Owned(layer_output.output.shallow_clone());
            all_layer_outputs.push(layer_output);
        }

        all_layer_outputs
    }

    fn n_layers(&self) -> i64 {
        self.layers.len() as i64
    }
}

#[derive(Debug)]
pub struct BertIntermediate {
    dense: Linear,
    activation: Box<dyn Module>,
}

impl BertIntermediate {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let activation = match bert_activations(&config.hidden_act) {
            Some(activation) => activation,
            None => return Err(BertError::unknown_activation_function(&config.hidden_act)),
        };

        Ok(BertIntermediate {
            activation,
            dense: bert_linear(
                vs / "dense",
                config,
                config.hidden_size,
                config.intermediate_size,
                "weight",
                "bias",
            ),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, input: &Tensor) -> Tensor {
        let hidden_states = self.dense.forward(input);
        self.activation.forward(&hidden_states)
    }
}

#[derive(Debug)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Result<Self, BertError> {
        let vs = vs.borrow();

        Ok(BertLayer {
            attention: BertAttention::new(vs / "attention", config),
            intermediate: BertIntermediate::new(vs / "intermediate", config)?,
            output: BertOutput::new(vs / "output", config),
        })
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> BertLayerOutput {
        let (attention_output, attention) = self.attention.forward_t(input, attention_mask, train);
        let intermediate_output = self.intermediate.forward(&attention_output);
        let output = self
            .output
            .forward_t(&intermediate_output, &attention_output, train);

        BertLayerOutput {
            output,
            attention: Some(attention),
        }
    }
}

/// Output of a BERT layer.
#[derive(Debug)]
pub struct BertLayerOutput {
    /// The output of the layer.
    pub output: Tensor,

    /// The layer attentions.
    pub attention: Option<Tensor>,
}

impl LayerAttention for BertLayerOutput {
    fn layer_attention(&self) -> Option<&Tensor> {
        self.attention.as_ref()
    }
}

impl LayerOutput for BertLayerOutput {
    fn layer_output(&self) -> &Tensor {
        &self.output
    }
}

#[derive(Debug)]
pub struct BertOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl BertOutput {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        let dense = bert_linear(
            vs / "dense",
            config,
            config.intermediate_size,
            config.hidden_size,
            "weight",
            "bias",
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let layer_norm = LayerNorm::new(
            vs / "layer_norm",
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );

        BertOutput {
            dense,
            dropout,
            layer_norm,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input: &Tensor, train: bool) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let mut hidden_states = self.dropout.forward_t(&hidden_states, train);
        hidden_states += input;
        self.layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug)]
pub struct BertSelfAttention {
    all_head_size: i64,
    attention_head_size: i64,
    num_attention_heads: i64,

    dropout: Dropout,
    key: Linear,
    query: Linear,
    value: Linear,
}

impl BertSelfAttention {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let key = bert_linear(
            vs / "key",
            config,
            config.hidden_size,
            all_head_size,
            "weight",
            "bias",
        );
        let query = bert_linear(
            vs / "query",
            config,
            config.hidden_size,
            all_head_size,
            "weight",
            "bias",
        );
        let value = bert_linear(
            vs / "value",
            config,
            config.hidden_size,
            all_head_size,
            "weight",
            "bias",
        );

        BertSelfAttention {
            all_head_size,
            attention_head_size,
            num_attention_heads: config.num_attention_heads,

            dropout: Dropout::new(config.attention_probs_dropout_prob),
            key,
            query,
            value,
        }
    }

    /// Apply self-attention.
    ///
    /// Return the contextualized representations and attention
    /// probabilities.
    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> (Tensor, Tensor) {
        let mixed_key_layer = self.key.forward(hidden_states);
        let mixed_query_layer = self.query.forward(hidden_states);
        let mixed_value_layer = self.value.forward(hidden_states);

        let query_layer = self.transpose_for_scores(&mixed_query_layer);
        let key_layer = self.transpose_for_scores(&mixed_key_layer);
        let value_layer = self.transpose_for_scores(&mixed_value_layer);

        // Get the raw attention scores.
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2));
        attention_scores /= (self.attention_head_size as f64).sqrt();

        if let Some(mask) = attention_mask {
            attention_scores += &**mask;
        }

        // Convert the raw attention scores into a probability distribution.
        let attention_probs = attention_scores.softmax(-1, Kind::Float);

        // Drop out entire tokens to attend to, following the original
        // transformer paper.
        let attention_probs = self.dropout.forward_t(&attention_probs, train);

        let context_layer = attention_probs.matmul(&value_layer);

        let context_layer = context_layer.permute(&[0, 2, 1, 3]).contiguous();
        let mut new_context_layer_shape = context_layer.size();
        new_context_layer_shape.splice(
            new_context_layer_shape.len() - 2..,
            iter::once(self.all_head_size),
        );
        let context_layer = context_layer.view_(&new_context_layer_shape);

        (context_layer, attention_probs)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let mut new_x_shape = x.size();
        new_x_shape.pop();
        new_x_shape.extend(&[self.num_attention_heads, self.attention_head_size]);

        x.view_(&new_x_shape).permute(&[0, 2, 1, 3])
    }
}

#[derive(Debug)]
pub struct BertSelfOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        let vs = vs.borrow();

        let dense = bert_linear(
            vs / "dense",
            config,
            config.hidden_size,
            config.hidden_size,
            "weight",
            "bias",
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let layer_norm = LayerNorm::new(
            vs / "layer_norm",
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );

        BertSelfOutput {
            dense,
            dropout,
            layer_norm,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input: &Tensor, train: bool) -> Tensor {
        let hidden_states = self.dense.forward(hidden_states);
        let mut hidden_states = self.dropout.forward_t(&hidden_states, train);
        hidden_states += input;
        self.layer_norm.forward(&hidden_states)
    }
}

fn bert_activations(activation_name: &str) -> Option<Box<dyn Module>> {
    match activation_name {
        "gelu" => Some(Box::new(activations::GELU)),
        "gelu_new" => Some(Box::new(activations::GELUNew)),
        _ => None,
    }
}

pub(crate) fn bert_linear<'a>(
    vs: impl Borrow<Path<'a>>,
    config: &BertConfig,
    in_features: i64,
    out_features: i64,
    weight_name: &str,
    bias_name: &str,
) -> Linear {
    let vs = vs.borrow();

    Linear {
        ws: vs.var(
            weight_name,
            &[out_features, in_features],
            Init::Randn {
                mean: 0.,
                stdev: config.initializer_range,
            },
        ),
        bs: vs.var(bias_name, &[out_features], Init::Const(0.)),
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BertError {
    #[cfg(feature = "load-hdf5")]
    #[error(transparent)]
    HDF5(#[from] hdf5::Error),

    #[error("hidden size ({hidden_size:?}) is not a multiple of attention heads ({num_attention_heads:?})")]
    IncorrectHiddenSize {
        hidden_size: i64,
        num_attention_heads: i64,
    },

    #[error("unknown activation function: {activation:?}")]
    UnknownActivationFunction { activation: String },
}

impl BertError {
    fn unknown_activation_function(activation: impl Into<String>) -> Self {
        BertError::UnknownActivationFunction {
            activation: activation.into(),
        }
    }
}

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::{Linear, Path};

    use crate::hdf5_model::{load_affine, load_tensor, LoadFromHDF5};
    use crate::layers::{Dropout, Embedding, LayerNorm, PlaceInVarStore};
    use crate::models::bert::{
        bert_activations, BertAttention, BertConfig, BertEmbeddings, BertEncoder, BertError,
        BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput,
    };

    impl LoadFromHDF5 for BertAttention {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            Ok(BertAttention {
                self_attention: BertSelfAttention::load_from_hdf5(
                    vs / "self",
                    config,
                    group.group("self")?,
                )?,
                self_output: BertSelfOutput::load_from_hdf5(
                    vs / "output",
                    config,
                    group.group("output")?,
                )?,
            })
        }
    }

    impl LoadFromHDF5 for BertEmbeddings {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            let word_embeddings = load_tensor(
                group.dataset("word_embeddings")?,
                &[config.vocab_size, config.hidden_size],
            )?;
            let position_embeddings = load_tensor(
                group.dataset("position_embeddings")?,
                &[config.max_position_embeddings, config.hidden_size],
            )?;
            let token_type_embeddings = load_tensor(
                group.dataset("token_type_embeddings")?,
                &[config.type_vocab_size, config.hidden_size],
            )?;

            let layer_norm_group = group.group("LayerNorm")?;

            let weight = load_tensor(layer_norm_group.dataset("weight")?, &[config.hidden_size])?;
            let bias = load_tensor(layer_norm_group.dataset("bias")?, &[config.hidden_size])?;

            Ok(BertEmbeddings {
                word_embeddings: Embedding(word_embeddings)
                    .place_in_var_store(vs / "word_embeddings"),
                position_embeddings: Embedding(position_embeddings)
                    .place_in_var_store(vs / "position_embeddings"),
                token_type_embeddings: Embedding(token_type_embeddings)
                    .place_in_var_store(vs / "token_type_embeddings"),

                layer_norm: LayerNorm::new_with_affine(
                    vec![config.hidden_size],
                    config.layer_norm_eps,
                    weight,
                    bias,
                )
                .place_in_var_store(vs / "layer_norm"),
                dropout: Dropout::new(config.hidden_dropout_prob),
            })
        }
    }

    impl LoadFromHDF5 for BertEncoder {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, BertError> {
            let vs = vs.borrow();

            let layers = (0..config.num_hidden_layers)
                .map(|idx| {
                    BertLayer::load_from_hdf5(
                        vs / format!("layer_{}", idx),
                        config,
                        group.group(&format!("layer_{}", idx))?,
                    )
                })
                .collect::<Result<_, _>>()?;

            Ok(BertEncoder { layers })
        }
    }

    impl LoadFromHDF5 for BertIntermediate {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let (dense_weight, dense_bias) = load_affine(
                group.group("dense")?,
                "weight",
                "bias",
                config.hidden_size,
                config.intermediate_size,
            )?;

            let activation = match bert_activations(&config.hidden_act) {
                Some(activation) => activation,
                None => return Err(BertError::unknown_activation_function(&config.hidden_act)),
            };

            Ok(BertIntermediate {
                activation,
                dense: Linear {
                    ws: dense_weight.tr(),
                    bs: dense_bias,
                }
                .place_in_var_store(vs.borrow() / "dense"),
            })
        }
    }

    impl LoadFromHDF5 for BertLayer {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, BertError> {
            let vs = vs.borrow();

            let attention =
                BertAttention::load_from_hdf5(vs / "attention", config, group.group("attention")?)?;
            let intermediate = BertIntermediate::load_from_hdf5(
                vs / "intermediate",
                config,
                group.group("intermediate")?,
            )?;

            let output = BertOutput::load_from_hdf5(vs / "output", config, group.group("output")?)?;

            Ok(BertLayer {
                attention,
                intermediate,
                output,
            })
        }
    }

    impl LoadFromHDF5 for BertOutput {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            let (dense_weight, dense_bias) = load_affine(
                group.group("dense")?,
                "weight",
                "bias",
                config.intermediate_size,
                config.hidden_size,
            )?;

            let layer_norm_group = group.group("LayerNorm")?;
            let layer_norm_weight =
                load_tensor(layer_norm_group.dataset("weight")?, &[config.hidden_size])?;
            let layer_norm_bias =
                load_tensor(layer_norm_group.dataset("bias")?, &[config.hidden_size])?;

            Ok(BertOutput {
                dense: Linear {
                    ws: dense_weight.tr(),
                    bs: dense_bias,
                }
                .place_in_var_store(vs / "dense"),
                dropout: Dropout::new(config.hidden_dropout_prob),
                layer_norm: LayerNorm::new_with_affine(
                    vec![config.hidden_size],
                    config.layer_norm_eps,
                    layer_norm_weight,
                    layer_norm_bias,
                )
                .place_in_var_store(vs / "layer_norm"),
            })
        }
    }

    #[cfg(feature = "load-hdf5")]
    impl LoadFromHDF5 for BertSelfAttention {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            let attention_head_size = config.hidden_size / config.num_attention_heads;
            let all_head_size = config.num_attention_heads * attention_head_size;

            let (key_weight, key_bias) = load_affine(
                group.group("key")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
            )?;
            let (query_weight, query_bias) = load_affine(
                group.group("query")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
            )?;
            let (value_weight, value_bias) = load_affine(
                group.group("value")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
            )?;

            Ok(BertSelfAttention {
                all_head_size,
                attention_head_size,
                num_attention_heads: config.num_attention_heads,

                dropout: Dropout::new(config.attention_probs_dropout_prob),
                key: Linear {
                    ws: key_weight.tr(),
                    bs: key_bias,
                }
                .place_in_var_store(vs / "key"),
                query: Linear {
                    ws: query_weight.tr(),
                    bs: query_bias,
                }
                .place_in_var_store(vs / "query"),
                value: Linear {
                    ws: value_weight.tr(),
                    bs: value_bias,
                }
                .place_in_var_store(vs / "value"),
            })
        }
    }

    impl LoadFromHDF5 for BertSelfOutput {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            let (dense_weight, dense_bias) = load_affine(
                group.group("dense")?,
                "weight",
                "bias",
                config.hidden_size,
                config.hidden_size,
            )?;

            let layer_norm_group = group.group("LayerNorm")?;
            let layer_norm_weight =
                load_tensor(layer_norm_group.dataset("weight")?, &[config.hidden_size])?;
            let layer_norm_bias =
                load_tensor(layer_norm_group.dataset("bias")?, &[config.hidden_size])?;

            Ok(BertSelfOutput {
                dense: Linear {
                    ws: dense_weight.tr(),
                    bs: dense_bias,
                }
                .place_in_var_store(vs / "dense"),
                dropout: Dropout::new(config.hidden_dropout_prob),
                layer_norm: LayerNorm::new_with_affine(
                    vec![config.hidden_size],
                    config.layer_norm_eps,
                    layer_norm_weight,
                    layer_norm_bias,
                )
                .place_in_var_store(vs / "layer_norm"),
            })
        }
    }
}

#[cfg(feature = "load-hdf5")]
#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use hdf5::File;
    use maplit::btreeset;
    use ndarray::{array, ArrayD};
    use tch::nn::{ModuleT, VarStore};
    use tch::{Device, Kind, Tensor};

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEmbeddings, BertEncoder, BertLayer};
    use crate::models::Encoder;

    const BERT_BASE_GERMAN_CASED: &str = env!("BERT_BASE_GERMAN_CASED");

    fn german_bert_config() -> BertConfig {
        BertConfig {
            attention_probs_dropout_prob: 0.1,
            hidden_act: "gelu".to_string(),
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

    fn layer_variables() -> BTreeSet<String> {
        btreeset![
            "attention.output.dense.bias".to_string(),
            "attention.output.dense.weight".to_string(),
            "attention.output.layer_norm.bias".to_string(),
            "attention.output.layer_norm.weight".to_string(),
            "attention.self.key.bias".to_string(),
            "attention.self.key.weight".to_string(),
            "attention.self.query.bias".to_string(),
            "attention.self.query.weight".to_string(),
            "attention.self.value.bias".to_string(),
            "attention.self.value.weight".to_string(),
            "intermediate.dense.bias".to_string(),
            "intermediate.dense.weight".to_string(),
            "output.dense.bias".to_string(),
            "output.dense.weight".to_string(),
            "output.layer_norm.bias".to_string(),
            "output.layer_norm.weight".to_string()
        ]
    }

    fn seqlen_to_mask(seq_lens: Tensor, max_len: i64) -> Tensor {
        let batch_size = seq_lens.size()[0];
        Tensor::arange(max_len, (Kind::Int, Device::Cpu))
            // Construct a matrix [batch_size, max_len] where each row
            // is 0..(max_len - 1).
            .repeat(&[batch_size])
            .view_(&[batch_size, max_len])
            // Time steps less than the length in seq_lens are active.
            .lt1(&seq_lens.unsqueeze(1))
    }

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn bert_embeddings() {
        let german_bert_config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &german_bert_config,
            german_bert_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let summed_embeddings =
            embeddings
                .forward_t(&pieces, false)
                .sum1(&[-1], false, Kind::Float);

        let sums: ArrayD<f32> = (&summed_embeddings).try_into().unwrap();

        // Verify output against Hugging Face transformers Python
        // implementation.
        assert_abs_diff_eq!(
            sums,
            (array![[
                -8.0342, -7.3383, -10.1286, 7.7298, 2.3506, -2.3831, -0.5961, -4.6270, -6.5415,
                2.1995
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_embeddings_names() {
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);
        BertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        let variables = varstore_variables(&vs);

        assert_eq!(
            variables,
            btreeset![
                "layer_norm.bias".to_string(),
                "layer_norm.weight".to_string(),
                "position_embeddings.embeddings".to_string(),
                "token_type_embeddings.embeddings".to_string(),
                "word_embeddings.embeddings".to_string()
            ]
        );

        // Compare against fresh embeddings layer.
        let vs_fresh = VarStore::new(Device::Cpu);
        let _ = BertEmbeddings::new(vs_fresh.root(), &config);
        assert_eq!(variables, varstore_variables(&vs_fresh));
    }

    #[test]
    fn bert_encoder() {
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("/bert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = BertEncoder::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/encoder").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let all_hidden_states = encoder.encode(&embeddings, None, false);

        let summed_last_hidden =
            all_hidden_states
                .last()
                .unwrap()
                .output
                .sum1(&[-1], false, Kind::Float);

        let sums: ArrayD<f32> = (&summed_last_hidden).try_into().unwrap();

        assert_abs_diff_eq!(
            sums,
            (array![[
                -1.6283, 0.2473, -0.2388, -0.4124, -0.4058, 1.4587, -0.3182, -0.9507, -0.1781,
                0.3792
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_encoder_attention_mask() {
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("/bert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = BertEncoder::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/encoder").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        // Add some padding to simulate inactive time steps.
        let pieces = Tensor::of_slice(&[
            133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2, 0, 0, 0, 0, 0,
        ])
        .reshape(&[1, 15]);

        let attention_mask = seqlen_to_mask(Tensor::of_slice(&[10]), pieces.size()[1]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let all_hidden_states = encoder.encode(&embeddings, Some(&attention_mask), false);

        let summed_last_hidden = all_hidden_states
            .last()
            .unwrap()
            .output
            .slice(-2, 0, 10, 1)
            .sum1(&[-1], false, Kind::Float);

        let sums: ArrayD<f32> = (&summed_last_hidden).try_into().unwrap();

        assert_abs_diff_eq!(
            sums,
            (array![[
                -1.6283, 0.2473, -0.2388, -0.4124, -0.4058, 1.4587, -0.3182, -0.9507, -0.1781,
                0.3792
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_encoder_names() {
        // Verify that the encoders's names correspond between loaded
        // and newly-constructed models.
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        BertEncoder::load_from_hdf5(
            vs_loaded.root(),
            &config,
            german_bert_file.group("bert/encoder").unwrap(),
        )
        .unwrap();
        let loaded_variables = varstore_variables(&vs_loaded);

        let mut encoder_variables = BTreeSet::new();
        let layer_variables = layer_variables();
        for idx in 0..config.num_hidden_layers {
            for layer_variable in &layer_variables {
                encoder_variables.insert(format!("layer_{}.{}", idx, layer_variable));
            }
        }

        assert_eq!(loaded_variables, encoder_variables);

        // Compare against fresh encoder.
        let vs_fresh = VarStore::new(Device::Cpu);
        let _ = BertEncoder::new(vs_fresh.root(), &config).unwrap();
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));
    }

    #[test]
    fn bert_layer() {
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        let layer0 = BertLayer::load_from_hdf5(
            vs.root(),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let layer_output0 = layer0.forward_t(&embeddings, None, false);

        let summed_layer0 = layer_output0.output.sum1(&[-1], false, Kind::Float);

        let sums: ArrayD<f32> = (&summed_layer0).try_into().unwrap();

        assert_abs_diff_eq!(
            sums,
            (array![[
                0.8649, -9.0162, -6.6015, 3.9470, -3.1475, -3.3533, -3.6431, -6.0901, -6.8157,
                -1.2723
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_layer_names() {
        // Verify that the layer's names correspond between loaded
        // and newly-constructed models.
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        BertLayer::load_from_hdf5(
            vs_loaded.root(),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();
        let loaded_variables = varstore_variables(&vs_loaded);

        let vs_fresh = VarStore::new(Device::Cpu);
        let _ = BertLayer::new(vs_fresh.root(), &config);

        assert_eq!(loaded_variables, layer_variables());
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));
    }
}
