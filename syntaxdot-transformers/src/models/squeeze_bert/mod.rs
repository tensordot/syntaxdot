//! SqueezeBERT (Iandola et al., 2020)
//!
//! SqueezeBERT follows the same architecture as BERT, but replaces most
//! matrix multiplications by grouped convolutions. This reduces the
//! number of parameters and speeds up inference.

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

use serde::{Deserialize, Serialize};
use tch::nn::{Module, ModuleT};
use tch::{Kind, Tensor};
use tch_ext::PathExt;

use crate::layers::{Conv1D, Dropout, LayerNorm};
use crate::models::bert::{bert_activations, BertError, BertLayerOutput};

use crate::models::{BertConfig, Encoder};
use crate::util::LogitsMask;

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
    ) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let activation = match bert_activations(activation) {
            Some(activation) => activation,
            None => return Err(BertError::unknown_activation_function(activation)),
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

        (context_layer, attention_probs)
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
    ) -> Result<Self, BertError> {
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
    ) -> BertLayerOutput {
        let (attention_output, attention) = self.attention.forward_t(input, attention_mask, train);
        let post_attention_output = self
            .post_attention
            .forward_t(&attention_output, input, train);
        let intermediate_output = self.intermediate.forward(&post_attention_output);
        let output = self
            .output
            .forward_t(&intermediate_output, &post_attention_output, train);

        BertLayerOutput {
            output,
            attention: Some(attention),
        }
    }
}

/// SqueezeBERT encoder.
///
/// Even though SqueezeBERT uses *[batch_size, hidden_size, seq_len]*
/// format internally, the encoder accepts the regular *[batch_size,
/// seq_len, hidden_size]* format.
#[derive(Debug)]
pub struct SqueezeBertEncoder {
    layers: Vec<SqueezeBertLayer>,
}

impl SqueezeBertEncoder {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &SqueezeBertConfig,
    ) -> Result<Self, BertError> {
        let vs = vs.borrow();

        let layers = (0..config.num_hidden_layers)
            .map(|layer| SqueezeBertLayer::new(vs / format!("layer_{}", layer), config))
            .collect::<Result<_, _>>()?;

        Ok(SqueezeBertEncoder { layers })
    }
}

impl Encoder for SqueezeBertEncoder {
    fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Vec<BertLayerOutput> {
        // [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]

        let mut all_layer_outputs = Vec::with_capacity(self.layers.len());

        let attention_mask = attention_mask.map(|mask| LogitsMask::from_bool_mask(mask));

        let mut hidden_states = input.permute(&[0, 2, 1]);

        all_layer_outputs.push(BertLayerOutput {
            output: hidden_states.shallow_clone(),
            attention: None,
        });

        for layer in &self.layers {
            let layer_output = layer.forward_t(&hidden_states, attention_mask.as_ref(), train);

            hidden_states = layer_output.output.shallow_clone();
            all_layer_outputs.push(layer_output);
        }

        // Convert hidden states to [batch_size, seq_len, hidden_size].
        for layer_output in &mut all_layer_outputs {
            layer_output.output = layer_output.output.permute(&[0, 2, 1]);
        }

        all_layer_outputs
    }

    fn n_layers(&self) -> i64 {
        self.layers.len() as i64 + 1
    }
}

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::ConvConfig;
    use tch_ext::PathExt;

    use crate::hdf5_model::{load_conv1d, load_tensor, LoadFromHDF5};
    use crate::layers::{Conv1D, Dropout, LayerNorm, PlaceInVarStore};

    use crate::models::bert::{bert_activations, BertError};
    use crate::models::squeeze_bert::{
        ConvActivation, ConvDropoutLayerNorm, SqueezeBertConfig, SqueezeBertEncoder,
        SqueezeBertLayer, SqueezeBertLayerNorm, SqueezeBertSelfAttention,
    };

    impl LoadFromHDF5 for SqueezeBertSelfAttention {
        type Config = SqueezeBertConfig;

        type Error = BertError;

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
        ) -> Result<Self, BertError> {
            let activation = match bert_activations(activation) {
                Some(activation) => activation,
                None => return Err(BertError::unknown_activation_function(activation)),
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
        ) -> Result<Self, BertError> {
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

        type Error = BertError;

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

    impl LoadFromHDF5 for SqueezeBertEncoder {
        type Config = SqueezeBertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, BertError> {
            let vs = vs.borrow();

            let layers = (0..config.num_hidden_layers)
                .map(|idx| {
                    SqueezeBertLayer::load_from_hdf5(
                        vs / format!("layer_{}", idx),
                        config,
                        group.group(&format!("layer_{}", idx))?,
                    )
                })
                .collect::<Result<_, _>>()?;

            Ok(SqueezeBertEncoder { layers })
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
    use tch_ext::RootExt;

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEmbeddings};
    use crate::models::squeeze_bert::{SqueezeBertConfig, SqueezeBertEncoder};
    use crate::models::Encoder;

    const SQUEEZEBERT_UNCASED: &str = env!("SQUEEZEBERT_UNCASED");

    fn squeezebert_uncased_config() -> SqueezeBertConfig {
        SqueezeBertConfig {
            attention_probs_dropout_prob: 0.1,
            embedding_size: 768,
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
            vocab_size: 30528,
            q_groups: 4,
            k_groups: 4,
            v_groups: 4,
            post_attention_groups: 1,
            intermediate_groups: 4,
            output_groups: 4,
        }
    }

    fn layer_variables() -> BTreeSet<String> {
        btreeset![
            "post_attention.conv1d.bias".to_string(),
            "post_attention.conv1d.weight".to_string(),
            "post_attention.layer_norm.bias".to_string(),
            "post_attention.layer_norm.weight".to_string(),
            "attention.key.bias".to_string(),
            "attention.key.weight".to_string(),
            "attention.query.bias".to_string(),
            "attention.query.weight".to_string(),
            "attention.value.bias".to_string(),
            "attention.value.weight".to_string(),
            "intermediate.conv1d.bias".to_string(),
            "intermediate.conv1d.weight".to_string(),
            "output.conv1d.bias".to_string(),
            "output.conv1d.weight".to_string(),
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
    fn squeeze_bert_embeddings() {
        let config = squeezebert_uncased_config();
        let bert_config: BertConfig = (&config).into();
        let file = File::open(SQUEEZEBERT_UNCASED).unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &bert_config,
            file.group("squeeze_bert/embeddings").unwrap(),
        )
        .unwrap();

        // Word pieces of: Did the AWO embezzle donations ?
        let pieces =
            Tensor::of_slice(&[2106i64, 1996, 22091, 2080, 7861, 4783, 17644, 11440, 1029])
                .reshape(&[1, 9]);

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
                39.4658, 35.4720, -2.2577, 11.3962, -1.6288, -9.8682, -18.4578, -12.0717, 11.7386
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn squeeze_bert_encoder() {
        let config = squeezebert_uncased_config();
        let bert_config: BertConfig = (&config).into();
        let file = File::open(SQUEEZEBERT_UNCASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &bert_config,
            file.group("/squeeze_bert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = SqueezeBertEncoder::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            file.group("squeeze_bert/encoder").unwrap(),
        )
        .unwrap();

        // Word pieces of: Did the AWO embezzle donations ?
        let pieces =
            Tensor::of_slice(&[2106i64, 1996, 22091, 2080, 7861, 4783, 17644, 11440, 1029])
                .reshape(&[1, 9]);

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
                -0.3894, -0.4608, -0.4127, -0.1656, -0.3927, -0.1952, -0.4998, -0.2477, -0.1676
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn squeeze_bert_encoder_attention_mask() {
        let config = squeezebert_uncased_config();
        let bert_config: BertConfig = (&config).into();
        let file = File::open(SQUEEZEBERT_UNCASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &bert_config,
            file.group("/squeeze_bert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = SqueezeBertEncoder::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            file.group("squeeze_bert/encoder").unwrap(),
        )
        .unwrap();

        // Word pieces of: Did the AWO embezzle donations ?
        // Add some padding to simulate inactive time steps.
        let pieces = Tensor::of_slice(&[
            2106i64, 1996, 22091, 2080, 7861, 4783, 17644, 11440, 1029, 0, 0, 0, 0, 0,
        ])
        .reshape(&[1, 14]);

        let attention_mask = seqlen_to_mask(Tensor::of_slice(&[9]), pieces.size()[1]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let all_hidden_states = encoder.encode(&embeddings, Some(&attention_mask), false);

        let summed_last_hidden = all_hidden_states
            .last()
            .unwrap()
            .output
            .slice(-2, 0, 9, 1)
            .sum1(&[-1], false, Kind::Float);

        let sums: ArrayD<f32> = (&summed_last_hidden).try_into().unwrap();

        assert_abs_diff_eq!(
            sums,
            (array![[
                -0.3894, -0.4608, -0.4127, -0.1656, -0.3927, -0.1952, -0.4998, -0.2477, -0.1676
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn bert_encoder_names_and_shapes() {
        // Verify that the encoders's names and shapes correspond between
        // loaded and newly-constructed models.
        let config = squeezebert_uncased_config();
        let file = File::open(SQUEEZEBERT_UNCASED).unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        SqueezeBertEncoder::load_from_hdf5(
            vs_loaded.root_ext(|_| 0),
            &config,
            file.group("squeeze_bert/encoder").unwrap(),
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
        let _ = SqueezeBertEncoder::new(vs_fresh.root_ext(|_| 0), &config).unwrap();
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));

        // Check shapes
        let loaded_variables = vs_loaded.variables();
        let fresh_variables = vs_fresh.variables();
        for (name, tensor) in loaded_variables {
            assert_eq!(tensor.size(), fresh_variables[&name].size());
        }
    }
}
