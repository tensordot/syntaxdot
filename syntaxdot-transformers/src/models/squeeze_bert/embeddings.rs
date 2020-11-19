#[cfg(feature = "load-hdf5")]
#[cfg(feature = "model-tests")]
#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use hdf5::File;
    use ndarray::{array, ArrayD};
    use syntaxdot_tch_ext::RootExt;
    use tch::nn::{ModuleT, VarStore};
    use tch::{Device, Kind, Tensor};

    use crate::hdf5_model::LoadFromHDF5;
    use crate::models::bert::{BertConfig, BertEmbeddings};
    use crate::models::squeeze_bert::SqueezeBertConfig;

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
}
