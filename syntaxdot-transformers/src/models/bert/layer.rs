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

use tch::nn::{Init, Linear, Module, ModuleT};
use tch::{Kind, Tensor};
use tch_ext::PathExt;

use crate::activations;
use crate::error::BertError;
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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &BertConfig) -> Result<Self, BertError> {
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

/// BERT layer
#[derive(Debug)]
pub struct BertLayer {
    attention: BertSelfAttention,
    post_attention: BertSelfOutput,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &BertConfig) -> Result<Self, BertError> {
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

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::Linear;
    use tch_ext::PathExt;

    use super::{
        bert_activations, BertIntermediate, BertLayer, BertOutput, BertSelfAttention,
        BertSelfOutput,
    };
    use crate::error::BertError;
    use crate::hdf5_model::{load_affine, load_tensor, LoadFromHDF5};
    use crate::layers::{Dropout, LayerNorm, PlaceInVarStore};
    use crate::models::bert::BertConfig;

    impl LoadFromHDF5 for BertIntermediate {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
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
            vs: impl Borrow<PathExt<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, BertError> {
            let vs = vs.borrow();
            let vs_attention = vs / "attention";
            let attention_group = group.group("attention")?;

            let attention = BertSelfAttention::load_from_hdf5(
                vs_attention.borrow() / "self",
                config,
                attention_group.group("self")?,
            )?;

            let post_attention = BertSelfOutput::load_from_hdf5(
                vs_attention.borrow() / "output",
                config,
                attention_group.group("output")?,
            )?;

            let intermediate = BertIntermediate::load_from_hdf5(
                vs / "intermediate",
                config,
                group.group("intermediate")?,
            )?;

            let output = BertOutput::load_from_hdf5(vs / "output", config, group.group("output")?)?;

            Ok(BertLayer {
                attention,
                post_attention,
                intermediate,
                output,
            })
        }
    }

    impl LoadFromHDF5 for BertOutput {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
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

    impl LoadFromHDF5 for BertSelfAttention {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
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
            vs: impl Borrow<PathExt<'a>>,
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
    use tch_ext::RootExt;

    use super::BertLayer;
    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEmbeddings};

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

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn bert_layer() {
        let config = german_bert_config();
        let german_bert_file = File::open(BERT_BASE_GERMAN_CASED).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = BertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            german_bert_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        let layer0 = BertLayer::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let layer_output0 = layer0.forward_t(&embeddings, None, false);

        let summed_layer0 = layer_output0.output().sum1(&[-1], false, Kind::Float);

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
            vs_loaded.root_ext(|_| 0),
            &config,
            german_bert_file.group("bert/encoder/layer_0").unwrap(),
        )
        .unwrap();
        let loaded_variables = varstore_variables(&vs_loaded);

        let vs_fresh = VarStore::new(Device::Cpu);
        let _ = BertLayer::new(vs_fresh.root_ext(|_| 0), &config);

        assert_eq!(loaded_variables, layer_variables());
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));
    }
}
