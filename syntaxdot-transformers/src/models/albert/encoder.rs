use std::borrow::Borrow;

use tch::nn::Module;
use tch::Tensor;
use tch_ext::PathExt;

use crate::error::TransformerError;
use crate::models::albert::{AlbertConfig, AlbertEmbeddingProjection};
use crate::models::bert::BertLayer;
use crate::models::layer_output::LayerOutput;
use crate::models::Encoder;
use crate::util::LogitsMask;

/// ALBERT encoder.
///
/// This encoder uses the BERT encoder with two modifications:
///
/// 1. The embeddings are projected to fit the hidden layer size.
/// 2. Weights are shared between layers.
#[derive(Debug)]
pub struct AlbertEncoder {
    groups: Vec<BertLayer>,
    n_layers: i64,
    projection: AlbertEmbeddingProjection,
}

impl AlbertEncoder {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &AlbertConfig,
    ) -> Result<Self, TransformerError> {
        assert!(
            config.num_hidden_groups > 0,
            "Need at least 1 hidden group, got: {}",
            config.num_hidden_groups
        );

        let vs = vs.borrow();

        let mut groups = Vec::with_capacity(config.num_hidden_groups as usize);
        for group_idx in 0..config.num_hidden_groups {
            groups.push(BertLayer::new(
                vs.sub(format!("group_{}", group_idx)).sub("inner_group_0"),
                &config.into(),
            )?);
        }
        let projection = AlbertEmbeddingProjection::new(vs, config);

        Ok(AlbertEncoder {
            groups,
            n_layers: config.num_hidden_layers,
            projection,
        })
    }
}

impl Encoder for AlbertEncoder {
    fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Vec<LayerOutput> {
        let mut all_layer_outputs = Vec::with_capacity(self.n_layers as usize + 1);

        let input = self.projection.forward(&input);

        all_layer_outputs.push(LayerOutput::Embedding(input.shallow_clone()));

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

        all_layer_outputs
    }

    fn n_layers(&self) -> i64 {
        self.n_layers + 1
    }
}

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch_ext::PathExt;

    use super::AlbertEncoder;
    use crate::error::TransformerError;
    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::albert::{AlbertConfig, AlbertEmbeddingProjection};
    use crate::models::bert::BertLayer;

    impl LoadFromHDF5 for AlbertEncoder {
        type Config = AlbertConfig;

        type Error = TransformerError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, TransformerError> {
            assert!(
                config.num_hidden_groups > 0,
                "Need at least 1 hidden group, got: {}",
                config.num_hidden_groups
            );

            let vs = vs.borrow();

            assert_eq!(
                config.inner_group_num, 1,
                "Only 1 inner group is supported, model has {}",
                config.inner_group_num
            );
            assert_eq!(
                config.num_hidden_groups, 1,
                "Only 1 hidden group is supported, model has {}",
                config.num_hidden_groups
            );

            let mut groups = Vec::with_capacity(config.num_hidden_groups as usize);
            for group_idx in 0..config.num_hidden_groups {
                groups.push(BertLayer::load_from_hdf5(
                    vs.sub(format!("group_{}", group_idx)).sub("inner_group_0"),
                    &config.into(),
                    group.group(&format!("group_{}/inner_group_0", group_idx))?,
                )?);
            }
            let projection = AlbertEmbeddingProjection::load_from_hdf5(vs, config, group)?;

            Ok(AlbertEncoder {
                groups,
                n_layers: config.num_hidden_layers,
                projection,
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

    use super::AlbertEncoder;
    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::albert::{AlbertConfig, AlbertEmbeddings};
    use crate::models::Encoder;

    const ALBERT_BASE_V2: &str = env!("ALBERT_BASE_V2");

    fn albert_config() -> AlbertConfig {
        AlbertConfig {
            attention_probs_dropout_prob: 0.,
            embedding_size: 128,
            hidden_act: "gelu_new".to_string(),
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
    fn albert_encoder() {
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = AlbertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            albert_file.group("albert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = AlbertEncoder::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            albert_file.group("albert/encoder").unwrap(),
        )
        .unwrap();

        // Pierre Vinken [...]
        let pieces = Tensor::of_slice(&[
            5399i64, 9730, 2853, 15, 6784, 122, 315, 15, 129, 1865, 14, 686, 9,
        ])
        .reshape(&[1, 13]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let all_hidden_states = encoder.encode(&embeddings, None, false);

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
                -19.8755, -22.0879, -22.1255, -22.1221, -22.1466, -21.9200, -21.7490, -22.4941,
                -21.7783, -21.9916, -21.5745, -22.1786, -21.9594
            ]])
            .into_dyn(),
            epsilon = 1e-3
        );
    }

    #[test]
    fn albert_encoder_attention_mask() {
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs = VarStore::new(Device::Cpu);

        let embeddings = AlbertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            albert_file.group("albert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = AlbertEncoder::load_from_hdf5(
            vs.root_ext(|_| 0),
            &config,
            albert_file.group("albert/encoder").unwrap(),
        )
        .unwrap();

        // Pierre Vinken [...]
        let pieces = Tensor::of_slice(&[
            5399i64, 9730, 2853, 15, 6784, 122, 315, 15, 129, 1865, 14, 686, 9, 0, 0,
        ])
        .reshape(&[1, 15]);

        let attention_mask = seqlen_to_mask(Tensor::of_slice(&[13]), pieces.size()[1]);

        let embeddings = embeddings.forward_t(&pieces, false);

        let all_hidden_states = encoder.encode(&embeddings, Some(&attention_mask), false);

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
                -19.8755, -22.0879, -22.1255, -22.1221, -22.1466, -21.9200, -21.7490, -22.4941,
                -21.7783, -21.9916, -21.5745, -22.1786, -21.9594, -21.7832, -21.7523
            ]])
            .into_dyn(),
            epsilon = 1e-3
        );
    }

    #[test]
    fn albert_encoder_names() {
        // Verify that the encoders's names correspond between loaded
        // and newly-constructed models.
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        AlbertEncoder::load_from_hdf5(
            vs_loaded.root_ext(|_| 0),
            &config,
            albert_file.group("albert/encoder").unwrap(),
        )
        .unwrap();
        let loaded_variables = varstore_variables(&vs_loaded);

        let mut encoder_variables = BTreeSet::new();
        let layer_variables = layer_variables();
        for layer_variable in &layer_variables {
            encoder_variables.insert(format!("group_0.inner_group_0.{}", layer_variable));
        }
        encoder_variables.insert("embedding_projection.weight".to_string());
        encoder_variables.insert("embedding_projection.bias".to_string());

        assert_eq!(loaded_variables, encoder_variables);

        // Compare against fresh encoder.
        let vs_fresh = VarStore::new(Device::Cpu);
        let _ = AlbertEncoder::new(vs_fresh.root_ext(|_| 0), &config).unwrap();
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));
    }
}