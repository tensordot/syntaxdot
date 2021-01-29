//! Word embeddings with sinusoidal position embeddings.

use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::Init;
use tch::{Kind, Tensor};

use crate::layers::{Dropout, Embedding, LayerNorm};
use crate::models::traits::WordEmbeddingsConfig;
use crate::module::{FallibleModule, FallibleModuleT};
use crate::util::SinusoidalPositions;
use crate::TransformerError;

/// Embeddings layer that uses word embeddings with sinusoidal positions.
///
/// The word embeddings in this layer can be optimized, but the sinusoidal
/// positions are generated on-the-fly.
#[derive(Debug)]
pub struct SinusoidalEmbeddings {
    dropout: Dropout,
    layer_norm: LayerNorm,
    p_norm: Option<f64>,
    word_embeddings: Embedding,
}

impl SinusoidalEmbeddings {
    /// Create piece embeddings with sinusoidal position embeddings.
    ///
    /// If a `p_norm` is specified position embeddings are normalized
    /// using this norm.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &impl WordEmbeddingsConfig,
        p_norm: Option<f64>,
    ) -> Result<SinusoidalEmbeddings, TransformerError> {
        let vs = vs.borrow();

        let normal_init = Init::Randn {
            mean: 0.,
            stdev: config.initializer_range(),
        };

        let word_embeddings = Embedding::new(
            vs / "word_embeddings",
            "embeddings",
            config.vocab_size(),
            config.dims(),
            normal_init,
        )?;

        let layer_norm = LayerNorm::new(
            vs / "layer_norm",
            vec![config.dims()],
            config.layer_norm_eps(),
            true,
        );

        let dropout = Dropout::new(config.dropout());

        Ok(SinusoidalEmbeddings {
            dropout,
            layer_norm,
            p_norm,
            word_embeddings,
        })
    }
}

impl FallibleModuleT for SinusoidalEmbeddings {
    type Error = TransformerError;

    fn forward_t(&self, input_ids: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        let word_embeddings = self.word_embeddings.forward(input_ids)?;

        let (_, seq_length, embedding_dim) = word_embeddings.size3()?;

        let position_embeddings: Tensor = SinusoidalPositions::sinusoidal_positions(
            seq_length,
            embedding_dim,
            self.p_norm,
            (Kind::Float, word_embeddings.device()),
        )?;

        let mut embeddings = tch::no_grad::<Result<_, TransformerError>, _>(|| {
            Ok(word_embeddings.f_add(&position_embeddings.f_unsqueeze(0)?)?)
        })?;
        embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward_t(&embeddings, train)
    }
}

#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use syntaxdot_tch_ext::RootExt;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};

    use crate::models::bert::BertConfig;
    use crate::models::sinusoidal::SinusoidalEmbeddings;
    use crate::module::FallibleModuleT;

    // BERT is not trained with sinusoidal embeddings, but we will just use
    // its piece embeddings to verify that the output of the
    // SinusoidalEmbeddings module hasn't changed.
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

    #[test]
    fn sinusoidal_embeddings_are_unchanged_without_norm() {
        let sums: ArrayD<f32> = get_and_sum_test_embeddings(None);

        // Verify output against known output (to avoid future breakage).
        assert_abs_diff_eq!(
            sums,
            (array![[
                -7.4332123, -7.3248596, -6.9817886, -5.2876663, -5.6577682, -6.173313, -6.041607,
                -6.035572, -5.6973915, -4.800396
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn sinusoidal_embeddings_are_unchanged_with_norm() {
        let sums: ArrayD<f32> = get_and_sum_test_embeddings(Some(2.0));

        // Verify output against known output (to avoid future breakage).
        assert_abs_diff_eq!(
            sums,
            (array![[
                -5.801262, -7.803936, -9.95359, 5.575783, 0.79592514, -3.6844482, -2.3470383,
                -5.6341896, -6.2476273, 1.965559
            ]])
            .into_dyn(),
            epsilon = 1e-4
        );
    }

    fn get_and_sum_test_embeddings(p_norm: Option<f64>) -> ArrayD<f32> {
        let config = german_bert_config();
        let mut vs = VarStore::new(Device::Cpu);
        let root = vs.root_ext(|_| 0);

        let embeddings =
            SinusoidalEmbeddings::new(root.sub("embeddings"), &config, p_norm).unwrap();

        vs.load(BERT_BASE_GERMAN_CASED).unwrap();

        // Word pieces of: Veruntreute die AWO spendengeld ?
        let pieces = Tensor::of_slice(&[133i64, 1937, 14010, 30, 32, 26939, 26962, 12558, 2739, 2])
            .reshape(&[1, 10]);

        let summed_embeddings =
            embeddings
                .forward_t(&pieces, false)
                .unwrap()
                .sum1(&[-1], false, Kind::Float);

        (&summed_embeddings).try_into().unwrap()
    }
}
