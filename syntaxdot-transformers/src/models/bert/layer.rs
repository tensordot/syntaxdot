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

use std::borrow::Borrow;
use std::iter;

use syntaxdot_tch_ext::PathExt;
use tch::nn::{Init, Linear, Module, ModuleT};
use tch::{Kind, Tensor};

use crate::activations;
use crate::error::TransformerError;
use crate::layers::{Dropout, LayerNorm};
use crate::models::bert::config::BertConfig;
use crate::models::layer_output::{HiddenLayer, LayerOutput};
use crate::util::LogitsMask;

#[derive(Debug)]
pub struct BertIntermediate {
    dense: Linear,
    activation: Box<dyn Module>,
}

impl BertIntermediate {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &BertConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        let activation = match bert_activations(&config.hidden_act) {
            Some(activation) => activation,
            None => {
                return Err(TransformerError::unknown_activation_function(
                    &config.hidden_act,
                ))
            }
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

/// BERT layer
#[derive(Debug)]
pub struct BertLayer {
    attention: BertSelfAttention,
    post_attention: BertSelfOutput,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &BertConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();
        let vs_attention = vs / "attention";

        Ok(BertLayer {
            attention: BertSelfAttention::new(vs_attention.borrow() / "self", config),
            post_attention: BertSelfOutput::new(vs_attention.borrow() / "output", config),
            intermediate: BertIntermediate::new(vs / "intermediate", config)?,
            output: BertOutput::new(vs / "output", config),
        })
    }

    pub fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> LayerOutput {
        let (attention_output, attention) = self.attention.forward_t(input, attention_mask, train);
        let post_attention_output = self
            .post_attention
            .forward_t(&attention_output, input, train);
        let intermediate_output = self.intermediate.forward(&post_attention_output);
        let output = self
            .output
            .forward_t(&intermediate_output, &post_attention_output, train);

        LayerOutput::EncoderWithAttention(HiddenLayer { output, attention })
    }
}

#[derive(Debug)]
pub struct BertOutput {
    dense: Linear,
    dropout: Dropout,
    layer_norm: LayerNorm,
}

impl BertOutput {
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &BertConfig) -> Self {
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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &BertConfig) -> Self {
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

        (context_layer, attention_scores)
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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &BertConfig) -> Self {
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

pub(crate) fn bert_activations(activation_name: &str) -> Option<Box<dyn Module>> {
    match activation_name {
        "gelu" => Some(Box::new(activations::GELU)),
        "gelu_new" => Some(Box::new(activations::GELUNew)),
        _ => None,
    }
}

pub(crate) fn bert_linear<'a>(
    vs: impl Borrow<PathExt<'a>>,
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
