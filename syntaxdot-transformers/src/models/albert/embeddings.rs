use std::borrow::Borrow;

use tch::nn::{Linear, Module, ModuleT};
use tch::Tensor;
use tch_ext::PathExt;

use crate::models::albert::AlbertConfig;
use crate::models::bert::bert_linear;
use crate::models::{BertConfig, BertEmbeddings};

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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &AlbertConfig) -> Self {
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
    pub fn new<'a>(vs: impl Borrow<PathExt<'a>>, config: &AlbertConfig) -> Self {
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

#[cfg(feature = "load-hdf5")]
mod hdf5_impl {
    use std::borrow::Borrow;

    use hdf5::Group;
    use tch::nn::Linear;
    use tch_ext::PathExt;

    use super::{AlbertEmbeddingProjection, AlbertEmbeddings};
    use crate::error::TransformerError;
    use crate::hdf5_model::{load_affine, LoadFromHDF5};
    use crate::layers::PlaceInVarStore;
    use crate::models::albert::AlbertConfig;
    use crate::models::bert::{BertConfig, BertEmbeddings};

    impl LoadFromHDF5 for AlbertEmbeddings {
        type Config = AlbertConfig;

        type Error = TransformerError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
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

        type Error = TransformerError;

        fn load_from_hdf5<'a>(
            vs: impl Borrow<PathExt<'a>>,
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
}

#[cfg(feature = "load-hdf5")]
#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hdf5::File;
    use maplit::btreeset;
    use tch::nn::VarStore;
    use tch::Device;
    use tch_ext::RootExt;

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::albert::{AlbertConfig, AlbertEmbeddings};

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

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test]
    fn albert_embeddings_names() {
        let config = albert_config();
        let albert_file = File::open(ALBERT_BASE_V2).unwrap();

        let vs = VarStore::new(Device::Cpu);
        AlbertEmbeddings::load_from_hdf5(
            vs.root_ext(|_| 0),
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
        let _ = AlbertEmbeddings::new(vs_fresh.root_ext(|_| 0), &config);
        assert_eq!(variables, varstore_variables(&vs_fresh));
    }
}
