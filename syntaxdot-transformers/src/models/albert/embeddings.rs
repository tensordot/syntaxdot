use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::{Linear, Module};
use tch::Tensor;

use crate::models::albert::AlbertConfig;
use crate::models::bert::{bert_linear, BertConfig, BertEmbeddings};
use crate::module::FallibleModuleT;
use crate::TransformerError;

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
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &AlbertConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        // BERT uses the hidden size as the vocab size.
        let mut bert_config: BertConfig = config.into();
        bert_config.hidden_size = config.embedding_size;

        let embeddings = BertEmbeddings::new(vs, &bert_config)?;

        Ok(AlbertEmbeddings { embeddings })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor, TransformerError> {
        self.embeddings
            .forward(input_ids, token_type_ids, position_ids, train)
    }
}

impl FallibleModuleT for AlbertEmbeddings {
    type Error = TransformerError;

    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        self.forward(input, None, None, train)
    }
}

/// Projection of ALBERT embeddings into the encoder hidden size.
#[derive(Debug)]
pub struct AlbertEmbeddingProjection {
    projection: Linear,
}

impl AlbertEmbeddingProjection {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &AlbertConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow();

        let projection = bert_linear(
            vs / "embedding_projection",
            &config.into(),
            config.embedding_size,
            config.hidden_size,
            "weight",
            "bias",
        )?;

        Ok(AlbertEmbeddingProjection { projection })
    }
}

impl Module for AlbertEmbeddingProjection {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.projection.forward(input)
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use maplit::btreeset;
    use syntaxdot_tch_ext::RootExt;
    use tch::nn::VarStore;
    use tch::Device;

    use crate::activations::Activation;
    use crate::models::albert::{AlbertConfig, AlbertEmbeddings};

    fn albert_config() -> AlbertConfig {
        AlbertConfig {
            attention_probs_dropout_prob: 0.,
            embedding_size: 128,
            hidden_act: Activation::GeluNew,
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

        let vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let _embeddings = AlbertEmbeddings::new(root, &config);

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
