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
use tch::nn::Init;
use tch::{Kind, Tensor};

use crate::cow::CowTensor;
use crate::layers::{Dropout, Embedding, LayerNorm};
use crate::models::bert::config::BertConfig;
use crate::module::{FallibleModule, FallibleModuleT};
use crate::TransformerError;

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
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &BertConfig,
    ) -> Result<Self, TransformerError> {
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
        )?;

        let position_embeddings = Embedding::new(
            vs / "position_embeddings",
            "embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            normal_init,
        )?;

        let token_type_embeddings = Embedding::new(
            vs / "token_type_embeddings",
            "embeddings",
            config.type_vocab_size,
            config.hidden_size,
            normal_init,
        )?;

        let layer_norm = LayerNorm::new(
            vs / "layer_norm",
            vec![config.hidden_size],
            config.layer_norm_eps,
            true,
        );

        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(BertEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, TransformerError> {
        let input_shape = input_ids.size();

        let seq_length = input_shape[1];
        let device = input_ids.device();

        let position_ids = match position_ids {
            Some(position_ids) => CowTensor::Borrowed(position_ids),
            None => CowTensor::Owned(
                Tensor::f_arange(seq_length, (Kind::Int64, device))?
                    .f_unsqueeze(0)?
                    // XXX: Second argument is 'implicit', do we need to set this?
                    .f_expand(&input_shape, false)?,
            ),
        };

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => CowTensor::Borrowed(token_type_ids),
            None => CowTensor::Owned(Tensor::f_zeros(&input_shape, (Kind::Int64, device))?),
        };

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&*position_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&*token_type_ids)?;

        let embeddings = input_embeddings
            .f_add(&position_embeddings)?
            .f_add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward_t(&embeddings, train)
    }
}

impl FallibleModuleT for BertEmbeddings {
    type Error = TransformerError;

    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        self.forward(input, None, None, train)
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

    use crate::models::bert::{BertConfig, BertEmbeddings};
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

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn bert_embeddings() {
        let config = german_bert_config();
        let mut vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let embeddings = BertEmbeddings::new(root.sub("embeddings"), &config).unwrap();

        vs.load(BERT_BASE_GERMAN_CASED).unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let summed_embeddings =
            embeddings
                .forward_t(&pieces, false)
                .unwrap()
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

        let vs = VarStore::new(Device::Cpu);
        let _ = BertEmbeddings::new(vs.root_ext(|_| 0), &config);

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
    }
}
