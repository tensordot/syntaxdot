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

use tch::nn::{Module, ModuleT};
use tch::{Kind, Tensor};
use tch_ext::PathExt;

use crate::error::TransformerError;
use crate::layers::{Conv1D, Dropout, LayerNorm};
use crate::models::bert::bert_activations;
use crate::models::layer_output::{HiddenLayer, LayerOutput};
use crate::models::squeeze_bert::SqueezeBertConfig;
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

impl Module for SqueezeBertLayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs_perm = xs.permute(&[0, 2, 1]);
        let xs_perm_norm = self.layer_norm.forward(&xs_perm);
        xs_perm_norm.permute(&[0, 2, 1])
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
    ) -> Self {
        let vs = vs.borrow();

        ConvDropoutLayerNorm {
            conv1d: Conv1D::new(vs / "conv1d", cin, cout, 1, groups),
            layer_norm: SqueezeBertLayerNorm::new(vs, cout, layer_norm_eps),
            dropout: Dropout::new(dropout_prob),
        }
    }

    fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let x = self.conv1d.forward(hidden_states);
        let x = self.dropout.forward_t(&x, train);
        let x = x + input_tensor;
        self.layer_norm.forward_t(&x, true)
    }
}

/// 1D convolution with an activation.
#[derive(Debug)]
struct ConvActivation {
    conv1d: Conv1D,
    activation: Box<dyn Module>,
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
            conv1d: Conv1D::new(vs.borrow() / "conv1d", cin, cout, 1, groups),
            activation,
        })
    }
}

impl Module for ConvActivation {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let output = self.conv1d.forward(&xs);
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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &SqueezeBertConfig) -> Self {
        let vs = vs.borrow();

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;

        let key = Conv1D::new(
            vs / "key",
            config.hidden_size,
            config.hidden_size,
            1,
            config.k_groups,
        );
        let query = Conv1D::new(
            vs / "query",
            config.hidden_size,
            config.hidden_size,
            1,
            config.q_groups,
        );
        let value = Conv1D::new(
            vs / "value",
            config.hidden_size,
            config.hidden_size,
            1,
            config.v_groups,
        );

        SqueezeBertSelfAttention {
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
    /// Return the contextualized representations and attention probabilities.
    ///
    /// Hidden states should be in *[batch_size, hidden_size, seq_len]* data
    /// layout.
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
        let key_layer = self.transpose_key_for_scores(&mixed_key_layer);
        let value_layer = self.transpose_for_scores(&mixed_value_layer);

        // Get the raw attention scores.
        let mut attention_scores = query_layer.matmul(&key_layer);
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

        let context_layer = self.transpose_output(&context_layer);

        (context_layer, attention_scores)
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let x_size = x.size();
        let new_x_shape = &[
            x_size[0],
            self.num_attention_heads,
            self.attention_head_size,
            *x_size.last().unwrap(),
        ];

        x.view_(new_x_shape).permute(&[0, 1, 3, 2])
    }

    fn transpose_key_for_scores(&self, x: &Tensor) -> Tensor {
        let x_size = x.size();
        let new_x_shape = &[
            x_size[0],
            self.num_attention_heads,
            self.attention_head_size,
            *x_size.last().unwrap(),
        ];

        x.view_(new_x_shape)
    }

    fn transpose_output(&self, x: &Tensor) -> Tensor {
        let x = x.permute(&[0, 1, 3, 2]).contiguous();
        let x_size = x.size();
        let new_x_shape = &[x_size[0], self.all_head_size, x_size[3]];
        x.view_(new_x_shape)
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
            attention: SqueezeBertSelfAttention::new(vs / "attention", config),
            post_attention: ConvDropoutLayerNorm::new(
                vs / "post_attention",
                config.hidden_size,
                config.hidden_size,
                config.post_attention_groups,
                config.hidden_dropout_prob,
                config.layer_norm_eps,
            ),
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
            ),
        })
    }
}

impl SqueezeBertLayer {
    /// Apply a SqueezeBERT layer.
    ///
    /// Hidden states should be in *[batch_size, hidden_size, seq_len]* data
    /// layout.
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

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::ConvConfig;
    use tch_ext::PathExt;

    use super::{
        ConvActivation, ConvDropoutLayerNorm, SqueezeBertLayer, SqueezeBertLayerNorm,
        SqueezeBertSelfAttention,
    };
    use crate::error::TransformerError;
    use crate::hdf5_model::{load_conv1d, load_tensor, LoadFromHDF5};
    use crate::layers::{Conv1D, Dropout, LayerNorm, PlaceInVarStore};
    use crate::models::bert::bert_activations;
    use crate::models::squeeze_bert::SqueezeBertConfig;

    impl LoadFromHDF5 for SqueezeBertSelfAttention {
        type Config = SqueezeBertConfig;

        type Error = TransformerError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            let attention_head_size = config.hidden_size / config.num_attention_heads;
            let all_head_size = config.num_attention_heads * attention_head_size;

            let (key_weight, key_bias) = load_conv1d(
                group.group("key")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
                1,
                config.k_groups,
            )?;
            let (query_weight, query_bias) = load_conv1d(
                group.group("query")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
                1,
                config.q_groups,
            )?;
            let (value_weight, value_bias) = load_conv1d(
                group.group("value")?,
                "weight",
                "bias",
                config.hidden_size,
                all_head_size,
                1,
                config.v_groups,
            )?;

            Ok(SqueezeBertSelfAttention {
                all_head_size,
                attention_head_size,
                num_attention_heads: config.num_attention_heads,

                dropout: Dropout::new(config.attention_probs_dropout_prob),
                key: Conv1D {
                    ws: key_weight,
                    bs: Some(key_bias),
                    config: ConvConfig {
                        groups: config.k_groups,
                        ..Default::default()
                    },
                }
                .place_in_var_store(vs / "key"),
                query: Conv1D {
                    ws: query_weight,
                    bs: Some(query_bias),
                    config: ConvConfig {
                        groups: config.q_groups,
                        ..Default::default()
                    },
                }
                .place_in_var_store(vs / "query"),
                value: Conv1D {
                    ws: value_weight,
                    bs: Some(value_bias),
                    config: ConvConfig {
                        groups: config.v_groups,
                        ..Default::default()
                    },
                }
                .place_in_var_store(vs / "value"),
            })
        }
    }

    impl ConvActivation {
        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            activation: &str,
            input_features: i64,
            output_features: i64,
            groups: i64,
            group: Group,
        ) -> Result<Self, TransformerError> {
            let activation = match bert_activations(activation) {
                Some(activation) => activation,
                None => return Err(TransformerError::unknown_activation_function(activation)),
            };

            // Fix: shapes are not always like this!
            let (conv_weight, conv_bias) = load_conv1d(
                group.group("conv1d")?,
                "weight",
                "bias",
                input_features,
                output_features,
                1,
                groups,
            )?;

            Ok(ConvActivation {
                conv1d: Conv1D {
                    ws: conv_weight,
                    bs: Some(conv_bias),
                    config: ConvConfig {
                        groups,
                        ..ConvConfig::default()
                    },
                }
                .place_in_var_store(vs.borrow() / "conv1d"),
                activation,
            })
        }
    }

    impl ConvDropoutLayerNorm {
        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            input_features: i64,
            output_features: i64,
            groups: i64,
            layer_norm_eps: f64,
            hidden_dropout_prob: f64,
            group: Group,
        ) -> Result<Self, TransformerError> {
            let vs = vs.borrow();

            let vs = vs.borrow();

            // Fix: shapes are not always like this!
            let (conv_weight, conv_bias) = load_conv1d(
                group.group("conv1d")?,
                "weight",
                "bias",
                input_features,
                output_features,
                1,
                groups,
            )?;

            let layer_norm_group = group.group("layernorm")?;
            let layer_norm_weight =
                load_tensor(layer_norm_group.dataset("weight")?, &[output_features])?;
            let layer_norm_bias =
                load_tensor(layer_norm_group.dataset("bias")?, &[output_features])?;

            Ok(ConvDropoutLayerNorm {
                conv1d: Conv1D {
                    ws: conv_weight,
                    bs: Some(conv_bias),
                    config: ConvConfig {
                        groups,
                        ..ConvConfig::default()
                    },
                }
                .place_in_var_store(vs / "conv1d"),
                layer_norm: SqueezeBertLayerNorm {
                    layer_norm: LayerNorm::new_with_affine(
                        vec![output_features],
                        layer_norm_eps,
                        layer_norm_weight,
                        layer_norm_bias,
                    )
                    .place_in_var_store(vs / "layer_norm"),
                },
                dropout: Dropout::new(hidden_dropout_prob),
            })
        }
    }

    impl LoadFromHDF5 for SqueezeBertLayer {
        type Config = SqueezeBertConfig;

        type Error = TransformerError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            config: &Self::Config,
            file: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            Ok(SqueezeBertLayer {
                attention: SqueezeBertSelfAttention::load_from_hdf5(
                    vs / "attention",
                    config,
                    file.group("attention")?,
                )?,
                post_attention: ConvDropoutLayerNorm::load_from_hdf5(
                    vs / "post_attention",
                    config.hidden_size,
                    config.hidden_size,
                    config.post_attention_groups,
                    config.layer_norm_eps,
                    config.hidden_dropout_prob,
                    file.group("post_attention")?,
                )?,
                intermediate: ConvActivation::load_from_hdf5(
                    vs / "intermediate",
                    &config.hidden_act,
                    config.hidden_size,
                    config.intermediate_size,
                    config.intermediate_groups,
                    file.group("intermediate")?,
                )?,
                output: ConvDropoutLayerNorm::load_from_hdf5(
                    vs / "output",
                    config.intermediate_size,
                    config.hidden_size,
                    config.output_groups,
                    config.layer_norm_eps,
                    config.hidden_dropout_prob,
                    file.group("output")?,
                )?,
            })
        }
    }
}
