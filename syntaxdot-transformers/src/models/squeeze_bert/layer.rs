// Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
// Copyright (c) 2020 TensorDot.
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

use syntaxdot_tch_ext::PathExt;
use tch::{Kind, Tensor};

use crate::error::TransformerError;
use crate::layers::{Conv1D, Dropout, LayerNorm};
use crate::models::bert::bert_activations;
use crate::models::layer_output::{HiddenLayer, LayerOutput};
use crate::models::squeeze_bert::SqueezeBertConfig;
use crate::module::{FallibleModule, FallibleModuleT};
use crate::util::LogitsMask;

/// Layer normalization for NCW data layout with normalization in C.
#[derive(Debug)]
pub struct SqueezeBertLayerNorm {
    layer_norm: LayerNorm,
}

impl SqueezeBertLayerNorm {
    fn new<'a>(vs: impl Borrow<PathExt<'a>>, hidden_size: i64, layer_norm_eps: f64) -> Self {
        SqueezeBertLayerNorm {
            layer_norm: LayerNorm::new(
                vs.borrow() / "layer_norm",
                vec![hidden_size],
                layer_norm_eps,
                true,
            ),
        }
    }
}

impl FallibleModule for SqueezeBertLayerNorm {
    type Error = TransformerError;

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Self::Error> {
        let xs_perm = xs.f_permute(&[0, 2, 1])?;
        let xs_perm_norm = self.layer_norm.forward(&xs_perm)?;
        Ok(xs_perm_norm.f_permute(&[0, 2, 1])?)
    }
}

/// Combined convolution, dropout, and layer normalization.
#[derive(Debug)]
struct ConvDropoutLayerNorm {
    conv1d: Conv1D,
    layer_norm: SqueezeBertLayerNorm,
    dropout: Dropout,
}

impl ConvDropoutLayerNorm {
    fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        cin: i64,
        cout: i64,
        groups: i64,
        dropout_prob: f64,
        layer_norm_eps: f64,
    ) -> Result<ConvDropoutLayerNorm, TransformerError> {
        let vs = vs.borrow();

        Ok(ConvDropoutLayerNorm {
            conv1d: Conv1D::new(vs / "conv1d", cin, cout, 1, groups)?,
            layer_norm: SqueezeBertLayerNorm::new(vs, cout, layer_norm_eps),
            dropout: Dropout::new(dropout_prob),
        })
    }

    fn forward_t(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor, TransformerError> {
        let x = self.conv1d.forward(hidden_states)?;
        let x = self.dropout.forward_t(&x, train)?;
        let x = x.f_add(&input_tensor)?;
        self.layer_norm.forward_t(&x, true)
    }
}

/// 1D convolution with an activation.
#[derive(Debug)]
struct ConvActivation {
    conv1d: Conv1D,
    activation: Box<dyn FallibleModule<Error = TransformerError>>,
}

impl ConvActivation {
    fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        cin: i64,
        cout: i64,
        groups: i64,
        activation: &str,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        let activation = match bert_activations(activation) {
            Some(activation) => activation,
            None => return Err(TransformerError::unknown_activation_function(activation)),
        };

        Ok(ConvActivation {
            conv1d: Conv1D::new(vs.borrow() / "conv1d", cin, cout, 1, groups)?,
            activation,
        })
    }
}

impl FallibleModule for ConvActivation {
    type Error = TransformerError;

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Self::Error> {
        let output = self.conv1d.forward(&xs)?;
        self.activation.forward(&output)
    }
}

/// Self-attention using grouped 1D convolutions.
#[derive(Debug)]
pub struct SqueezeBertSelfAttention {
    all_head_size: i64,
    attention_head_size: i64,
    num_attention_heads: i64,

    dropout: Dropout,
    key: Conv1D,
    query: Conv1D,
    value: Conv1D,
}

impl SqueezeBertSelfAttention {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &SqueezeBertConfig,
    ) -> Result<SqueezeBertSelfAttention, TransformerError> {
        let vs = vs.borrow();

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let key = Conv1D::new(
            vs / "key",
            config.hidden_size,
            config.hidden_size,
            1,
            config.k_groups,
        )?;
        let query = Conv1D::new(
            vs / "query",
            config.hidden_size,
            config.hidden_size,
            1,
            config.q_groups,
        )?;
        let value = Conv1D::new(
            vs / "value",
            config.hidden_size,
            config.hidden_size,
            1,
            config.v_groups,
        )?;

        Ok(SqueezeBertSelfAttention {
            all_head_size,
            attention_head_size,
            num_attention_heads: config.num_attention_heads,

            dropout: Dropout::new(config.attention_probs_dropout_prob),
            key,
            query,
            value,
        })
    }

    /// Apply self-attention.
    ///
    /// Return the contextualized representations and attention probabilities.
    ///
    /// Hidden states should be in *[batch_size, hidden_size, seq_len]* data
    /// layout.
    fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> Result<(Tensor, Tensor), TransformerError> {
        let mixed_key_layer = self.key.forward(hidden_states)?;
        let mixed_query_layer = self.query.forward(hidden_states)?;
        let mixed_value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&mixed_query_layer)?;
        let key_layer = self.transpose_key_for_scores(&mixed_key_layer)?;
        let value_layer = self.transpose_for_scores(&mixed_value_layer)?;

        // Get the raw attention scores.
        let mut attention_scores = query_layer.f_matmul(&key_layer)?;
        let _ = attention_scores.f_div_1((self.attention_head_size as f64).sqrt());

        if let Some(mask) = attention_mask {
            let _ = attention_scores.f_add_(&**mask)?;
        }

        // Convert the raw attention scores into a probability distribution.
        let attention_probs = attention_scores.f_softmax(-1, Kind::Float)?;

        // Drop out entire tokens to attend to, following the original
        // transformer paper.
        let attention_probs = self.dropout.forward_t(&attention_probs, train)?;

        let context_layer = attention_probs.f_matmul(&value_layer)?;

        let context_layer = self.transpose_output(&context_layer)?;

        Ok((context_layer, attention_scores))
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor, TransformerError> {
        let x_size = x.size();
        let new_x_shape = &[
            x_size[0],
            self.num_attention_heads,
            self.attention_head_size,
            *x_size.last().unwrap(),
        ];

        Ok(x.f_view_(new_x_shape)?.f_permute(&[0, 1, 3, 2])?)
    }

    fn transpose_key_for_scores(&self, x: &Tensor) -> Result<Tensor, TransformerError> {
        let x_size = x.size();
        let new_x_shape = &[
            x_size[0],
            self.num_attention_heads,
            self.attention_head_size,
            *x_size.last().unwrap(),
        ];

        Ok(x.f_view_(new_x_shape)?)
    }

    fn transpose_output(&self, x: &Tensor) -> Result<Tensor, TransformerError> {
        let x = x.f_permute(&[0, 1, 3, 2])?.f_contiguous()?;
        let x_size = x.size();
        let new_x_shape = &[x_size[0], self.all_head_size, x_size[3]];
        Ok(x.f_view_(new_x_shape)?)
    }
}

/// SqueezeBERT layer.
#[derive(Debug)]
pub struct SqueezeBertLayer {
    attention: SqueezeBertSelfAttention,
    post_attention: ConvDropoutLayerNorm,
    intermediate: ConvActivation,
    output: ConvDropoutLayerNorm,
}

impl SqueezeBertLayer {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &SqueezeBertConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        Ok(SqueezeBertLayer {
            attention: SqueezeBertSelfAttention::new(vs / "attention", config)?,
            post_attention: ConvDropoutLayerNorm::new(
                vs / "post_attention",
                config.hidden_size,
                config.hidden_size,
                config.post_attention_groups,
                config.hidden_dropout_prob,
                config.layer_norm_eps,
            )?,
            intermediate: ConvActivation::new(
                vs / "intermediate",
                config.hidden_size,
                config.intermediate_size,
                config.intermediate_groups,
                &config.hidden_act,
            )?,
            output: ConvDropoutLayerNorm::new(
                vs / "output",
                config.intermediate_size,
                config.hidden_size,
                config.output_groups,
                config.hidden_dropout_prob,
                config.layer_norm_eps,
            )?,
        })
    }
}

impl SqueezeBertLayer {
    /// Apply a SqueezeBERT layer.
    ///
    /// Hidden states should be in *[batch_size, hidden_size, seq_len]* data
    /// layout.
    pub(crate) fn forward_t(
        &self,
        input: &Tensor,
        attention_mask: Option<&LogitsMask>,
        train: bool,
    ) -> Result<LayerOutput, TransformerError> {
        let (attention_output, attention) =
            self.attention.forward_t(input, attention_mask, train)?;
        let post_attention_output =
            self.post_attention
                .forward_t(&attention_output, input, train)?;
        let intermediate_output = self.intermediate.forward(&post_attention_output)?;
        let output = self
            .output
            .forward_t(&intermediate_output, &post_attention_output, train)?;

        Ok(LayerOutput::EncoderWithAttention(HiddenLayer {
            output,
            attention,
        }))
    }
}
