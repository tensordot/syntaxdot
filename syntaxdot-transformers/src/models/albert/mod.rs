//! ALBERT (Lan et al., 2020)

use std::borrow::Borrow;

use serde::{Deserialize, Serialize};
use tch::nn::{Linear, Module, ModuleT, Path};
use tch::Tensor;

use crate::models::bert::{
    bert_linear, BertConfig, BertEmbeddings, BertError, BertLayer, BertLayerOutput,
};
use crate::models::traits::WordEmbeddingsConfig;
use crate::models::Encoder;
use crate::util::LogitsMask;

/// ALBERT model configuration.
#[serde(default)]
#[derive(Debug, Deserialize, Serialize)]
pub struct AlbertConfig {
    pub attention_probs_dropout_prob: f64,
    pub embedding_size: i64,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub inner_group_num: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_groups: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        AlbertConfig {
            attention_probs_dropout_prob: 0.,
            embedding_size: 128,
            hidden_act: "gelu_new".to_owned(),
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
}

impl From<&AlbertConfig> for BertConfig {
    fn from(albert_config: &AlbertConfig) -> Self {
        BertConfig {
            attention_probs_dropout_prob: albert_config.attention_probs_dropout_prob,
            hidden_act: albert_config.hidden_act.clone(),
            hidden_dropout_prob: albert_config.hidden_dropout_prob,
            hidden_size: albert_config.hidden_size,
            initializer_range: albert_config.initializer_range,
            intermediate_size: albert_config.intermediate_size,
            layer_norm_eps: 1e-12,
            max_position_embeddings: albert_config.max_position_embeddings,
            num_attention_heads: albert_config.num_attention_heads,
            num_hidden_layers: albert_config.num_hidden_layers,
            type_vocab_size: albert_config.type_vocab_size,
            vocab_size: albert_config.vocab_size,
        }
    }
}

impl WordEmbeddingsConfig for AlbertConfig {
    fn dims(&self) -> i64 {
        self.embedding_size
    }

    fn dropout(&self) -> f64 {
        self.hidden_dropout_prob
    }

    fn initializer_range(&self) -> f64 {
        self.initializer_range
    }

    fn layer_norm_eps(&self) -> f64 {
        1e-12
    }

    fn vocab_size(&self) -> i64 {
        self.vocab_size
    }
}

/// ALBERT embeddings.
///
/// These embeddings are the same as BERT embeddings. However, we do
/// some wrapping to ensure that the right embedding dimensionality is
/// used.
#[derive(Debug)]
pub struct AlbertEmbeddings {
    embeddings: BertEmbeddings,
}

impl AlbertEmbeddings {
    /// Construct new ALBERT embeddings with the given variable store
    /// and ALBERT configuration.
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &AlbertConfig) -> Self {
        let vs = vs.borrow();

        // BERT uses the hidden size as the vocab size.
        let mut bert_config: BertConfig = config.into();
        bert_config.hidden_size = config.embedding_size;

        let embeddings = BertEmbeddings::new(vs, &bert_config);

        AlbertEmbeddings { embeddings }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        self.embeddings
            .forward(input_ids, token_type_ids, position_ids, train)
    }
}

impl ModuleT for AlbertEmbeddings {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        self.forward(input, None, None, train)
    }
}

/// Projection of ALBERT embeddings into the encoder hidden size.
#[derive(Debug)]
pub struct AlbertEmbeddingProjection {
    projection: Linear,
}

impl AlbertEmbeddingProjection {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &AlbertConfig) -> Self {
        let vs = vs.borrow();

        let projection = bert_linear(
            vs / "embedding_projection",
            &config.into(),
            config.embedding_size,
            config.hidden_size,
            "weight",
            "bias",
        );

        AlbertEmbeddingProjection { projection }
    }
}

impl Module for AlbertEmbeddingProjection {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.projection.forward(input)
    }
}

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
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &AlbertConfig) -> Result<Self, BertError> {
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
    ) -> Vec<BertLayerOutput> {
        let mut all_layer_outputs = Vec::with_capacity(self.n_layers as usize + 1);

        let input = self.projection.forward(&input);

        all_layer_outputs.push(BertLayerOutput {
            output: input.shallow_clone(),
            attention: None,
        });

        let attention_mask = attention_mask.map(|mask| LogitsMask::from_bool_mask(mask));

        let layers_per_group = self.n_layers as usize / self.groups.len();

        let mut hidden_states = input;
        for idx in 0..self.n_layers {
            let layer_output = self.groups[idx as usize / layers_per_group].forward_t(
                &hidden_states,
                attention_mask.as_ref(),
                train,
            );

            hidden_states = layer_output.output.shallow_clone();

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
    use tch::nn::{Linear, Path};

    use crate::hdf5_model::{load_affine, LoadFromHDF5};
    use crate::layers::PlaceInVarStore;
    use crate::models::albert::{
        AlbertConfig, AlbertEmbeddingProjection, AlbertEmbeddings, AlbertEncoder,
    };
    use crate::models::bert::{BertConfig, BertEmbeddings, BertError, BertLayer};

    impl LoadFromHDF5 for AlbertEmbeddings {
        type Config = AlbertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let vs = vs.borrow();

            // BERT uses the hidden size as the vocab size.
            let mut bert_config: BertConfig = config.into();
            bert_config.hidden_size = config.embedding_size;

            let embeddings = BertEmbeddings::load_from_hdf5(vs, &bert_config, group)?;

            Ok(AlbertEmbeddings { embeddings })
        }
    }

    impl LoadFromHDF5 for AlbertEmbeddingProjection {
        type Config = AlbertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, Self::Error> {
            let (dense_weight, dense_bias) = load_affine(
                group.group("embedding_projection")?,
                "weight",
                "bias",
                config.embedding_size,
                config.hidden_size,
            )?;

            Ok(AlbertEmbeddingProjection {
                projection: Linear {
                    ws: dense_weight.tr(),
                    bs: dense_bias,
                }
                .place_in_var_store(vs.borrow() / "embedding_projection"),
            })
        }
    }

    impl LoadFromHDF5 for AlbertEncoder {
        type Config = AlbertConfig;

        type Error = BertError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<Path<'a>>,
            config: &Self::Config,
            group: Group,
        ) -> Result<Self, BertError> {
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

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::albert::{AlbertConfig, AlbertEmbeddings, AlbertEncoder};
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
            vs.root(),
            &config,
            albert_file.group("albert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = AlbertEncoder::load_from_hdf5(
            vs.root(),
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
                .output
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
            vs.root(),
            &config,
            albert_file.group("albert/embeddings").unwrap(),
        )
        .unwrap();

        let encoder = AlbertEncoder::load_from_hdf5(
            vs.root(),
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
                .output
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
    fn albert_embeddings_names() {
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs = VarStore::new(Device::Cpu);
        AlbertEmbeddings::load_from_hdf5(
            vs.root(),
            &config,
            albert_file.group("albert/embeddings").unwrap(),
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
        let _ = AlbertEmbeddings::new(vs_fresh.root(), &config);
        assert_eq!(variables, varstore_variables(&vs_fresh));
    }

    #[test]
    fn albert_encoder_names() {
        // Verify that the encoders's names correspond between loaded
        // and newly-constructed models.
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs_loaded = VarStore::new(Device::Cpu);
        AlbertEncoder::load_from_hdf5(
            vs_loaded.root(),
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
        let _ = AlbertEncoder::new(vs_fresh.root(), &config).unwrap();
        assert_eq!(loaded_variables, varstore_variables(&vs_fresh));
    }
}
