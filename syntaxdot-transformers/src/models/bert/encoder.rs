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

use syntaxdot_tch_ext::PathExt;
use tch::Tensor;

use crate::cow::CowTensor;
use crate::error::TransformerError;
use crate::models::bert::{BertConfig, BertLayer};
use crate::models::layer_output::LayerOutput;
use crate::models::Encoder;
use crate::util::LogitsMask;

/// BERT encoder.
#[derive(Debug)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &BertConfig,
    ) -> Result<Self, TransformerError> {
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
    ) -> Result<Vec<LayerOutput>, TransformerError> {
        let mut all_layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        all_layer_outputs.push(LayerOutput::Embedding(input.shallow_clone()));

        let attention_mask = attention_mask
            .map(|mask| LogitsMask::from_bool_mask(mask))
            .transpose()?;

        let mut hidden_states = CowTensor::Borrowed(input);
        for layer in &self.layers {
            let layer_output = layer.forward_t(&hidden_states, attention_mask.as_ref(), train)?;

            hidden_states = CowTensor::Owned(layer_output.output().shallow_clone());
            all_layer_outputs.push(layer_output);
        }

        Ok(all_layer_outputs)
    }

    fn n_layers(&self) -> i64 {
        self.layers.len() as i64 + 1
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use maplit::btreeset;
    use ndarray::{array, ArrayD};
    use syntaxdot_tch_ext::RootExt;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use crate::models::bert::{BertConfig, BertEmbeddings, BertEncoder};
    use crate::models::Encoder;
    use crate::module::FallibleModuleT;

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
    fn bert_encoder() {
        let config = german_bert_config();

        let mut vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let embeddings = BertEmbeddings::new(root.sub("embeddings"), &config).unwrap();
        let encoder = BertEncoder::new(root.sub("encoder"), &config).unwrap();

        vs.load(BERT_BASE_GERMAN_CASED).unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let embeddings = embeddings.forward_t(&pieces, false).unwrap();

        let all_hidden_states = encoder.encode(&embeddings, None, false).unwrap();

        let summed_last_hidden =
            all_hidden_states
                .last()
                .unwrap()
                .output()
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

        let mut vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let embeddings = BertEmbeddings::new(root.sub("embeddings"), &config).unwrap();
        let encoder = BertEncoder::new(root.sub("encoder"), &config).unwrap();

        vs.load(BERT_BASE_GERMAN_CASED).unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        // Add some padding to simulate inactive time steps.
        let pieces = Tensor::of_slice(&[
            133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2, 0, 0, 0, 0, 0,
        ])
        .reshape(&[1, 15]);

        let attention_mask = seqlen_to_mask(Tensor::of_slice(&[10]), pieces.size()[1]);

        let embeddings = embeddings.forward_t(&pieces, false).unwrap();

        let all_hidden_states = encoder
            .encode(&embeddings, Some(&attention_mask), false)
            .unwrap();

        let summed_last_hidden = all_hidden_states
            .last()
            .unwrap()
            .output()
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
        // Verify that the encoders's names are correct.
        // and newly-constructed models.
        let config = german_bert_config();

        let vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let _encoder = BertEncoder::new(root, &config).unwrap();

        let mut encoder_variables = BTreeSet::new();
        let layer_variables = layer_variables();
        for idx in 0..config.num_hidden_layers {
            for layer_variable in &layer_variables {
                encoder_variables.insert(format!("layer_{}.{}", idx, layer_variable));
            }
        }

        assert_eq!(varstore_variables(&vs), encoder_variables);
    }
}
