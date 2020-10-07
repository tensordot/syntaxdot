// Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2020 The sticker developers.
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

use tch::nn::{ModuleT, Path};
use tch::{Kind, Tensor};

use crate::cow::CowTensor;
use crate::models::bert::{BertConfig, BertEmbeddings};

const PADDING_IDX: i64 = 1;

/// RoBERTa and XLM-RoBERTa embeddings.
#[derive(Debug)]
pub struct RobertaEmbeddings {
    inner: BertEmbeddings,
}

impl RobertaEmbeddings {
    /// Construct new RoBERTa embeddings with the given variable store
    /// and Bert configuration.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &BertConfig) -> Self {
        RobertaEmbeddings {
            inner: BertEmbeddings::new(vs, config),
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let position_ids = match position_ids {
            Some(position_ids) => CowTensor::Borrowed(position_ids),
            None => {
                let mask = input_ids.ne(PADDING_IDX).to_kind(Kind::Int64);
                let incremental_indices = mask.cumsum(1, Kind::Int64) * mask;
                CowTensor::Owned(incremental_indices + PADDING_IDX)
            }
        };

        self.inner.forward(
            input_ids,
            token_type_ids,
            Some(position_ids.as_ref()),
            train,
        )
    }
}

impl ModuleT for RobertaEmbeddings {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.forward(input, None, None, train)
    }
}

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::Path;

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEmbeddings, BertError};
    use crate::models::roberta::RobertaEmbeddings;

    impl LoadFromHDF5 for RobertaEmbeddings {
        type Config = BertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            file: Group,
        ) -> Result<Self, Self::Error> {
            BertEmbeddings::load_from_hdf5(vs, config, file)
                .map(|embeds| RobertaEmbeddings { inner: embeds })
        }
    }
}

#[cfg(feature = "model-tests")]
#[cfg(feature = "load-hdf5")]
#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use hdf5::File;
    use ndarray::{array, ArrayD};
    use tch::nn::{ModuleT, VarStore};
    use tch::{Device, Kind, Tensor};

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEncoder};
    use crate::models::roberta::RobertaEmbeddings;
    use crate::models::Encoder;

    const XLM_ROBERTA_BASE: &str = env!("XLM_ROBERTA_BASE");

    fn xlm_roberta_config() -> BertConfig {
        BertConfig {
            attention_probs_dropout_prob: 0.1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            initializer_range: 0.02,
            intermediate_size: 3072,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 514,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            type_vocab_size: 1,
            vocab_size: 250002,
        }
    }

    #[test]
    fn xlm_roberta_embeddings() {
        let config = xlm_roberta_config();
        let roberta_file = File::open(XLM_ROBERTA_BASE).unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings = RobertaEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            roberta_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        // Subtokenization of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[
            0i64, 310, 23451, 107, 6743, 68, 62, 43789, 207126, 49004, 705, 2,
        ])
        .reshape(&[1, 12]);

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
                -9.1686, -4.2982, -0.7808, -0.7097, 0.0972, -3.0785, -3.6755, -2.1465, -2.9406,
                -1.0627, -6.6043, -4.8064
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn xlm_roberta_encoder() {
        let config = xlm_roberta_config();
        let roberta_file = File::open(XLM_ROBERTA_BASE).unwrap();

        let vs = VarStore::new(Device::Cpu);
        let embeddings = RobertaEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            roberta_file.group("bert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = BertEncoder::load_from_hdf5(
            vs.root(),
            &config,
            roberta_file.group("bert/encoder").unwrap(),
        )
        .unwrap();

        // Subtokenization of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[
            0i64, 310, 23451, 107, 6743, 68, 62, 43789, 207126, 49004, 705, 2,
        ])
        .reshape(&[1, 12]);

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
                20.9693, 19.7502, 17.0594, 19.0700, 19.0065, 19.6254, 18.9379, 18.9275, 18.8922,
                18.9505, 19.2682, 20.9411
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }
}
