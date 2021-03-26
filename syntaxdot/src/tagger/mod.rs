use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::convert::TryInto;

use ndarray::{s, Array1, ArrayD, Axis};
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_encoders::{EncodingProb, SentenceDecoder};
use syntaxdot_tokenizers::SentenceWithPieces;
use tch::Device;

use crate::encoders::Encoders;
use crate::error::SyntaxDotError;
use crate::model::bert::BertModel;
use crate::model::biaffine_dependency_layer::BiaffineScoreLogits;
use crate::model::seq_classifiers::TopK;
use crate::tensor::{TensorBuilder, Tensors};

/// A sequence tagger.
pub struct Tagger {
    biaffine_encoder: Option<ImmutableDependencyEncoder>,
    device: Device,
    encoders: Encoders,
    model: BertModel,
}

impl Tagger {
    /// Construct a new tagger.
    pub fn new(
        device: Device,
        model: BertModel,
        biaffine_encoder: Option<ImmutableDependencyEncoder>,
        encoders: Encoders,
    ) -> Self {
        Tagger {
            device,
            model,
            biaffine_encoder,
            encoders,
        }
    }

    /// Tag sentences.
    pub fn tag_sentences(
        &self,
        sentences: &mut [impl BorrowMut<SentenceWithPieces>],
    ) -> Result<(), SyntaxDotError> {
        let tensors = self.prepare_batch(sentences);

        // Get model predictions.
        let attention_mask = tensors.seq_lens.attention_mask()?;
        let predictions = self.model.predict(
            &tensors.inputs.to_device(self.device),
            &attention_mask.to_device(self.device),
            &tensors.token_spans.to_device(self.device),
        )?;

        assert_eq!(
            self.biaffine_encoder.is_some(),
            predictions.biaffine_score_logits.is_some(),
            "Biaffine encoder and predictions should both be present (or absent), was: {} {}",
            self.biaffine_encoder.is_some(),
            predictions.biaffine_score_logits.is_some(),
        );

        // Decode dependencies before sequence labels. Biaffine parsing does not require any
        // other annotations. Sequence labelers, however, may require dependencies (e.g. the
        // TüBa-D/Z lemmatizer).
        if let (Some(encoder), Some(biaffine_score_logits)) = (
            self.biaffine_encoder.as_ref(),
            predictions.biaffine_score_logits,
        ) {
            tch::no_grad(|| self.decode_biaffine(encoder, sentences, biaffine_score_logits))?
        }

        self.decode_sequence_labels(sentences, predictions.sequences_top_k)?;

        Ok(())
    }

    /// Construct the tensor representations of a batch of sentences.
    fn prepare_batch(&self, sentences: &[impl Borrow<SentenceWithPieces>]) -> Tensors {
        let max_seq_len = sentences
            .iter()
            .map(|sentence| sentence.borrow().pieces.len())
            .max()
            .unwrap_or(0);

        let max_tokens_len = sentences
            .iter()
            .map(|sentence| sentence.borrow().token_offsets.len())
            .max()
            .unwrap_or(0);

        let mut builder: TensorBuilder =
            TensorBuilder::new_without_labels(sentences.len(), max_seq_len, max_tokens_len);

        for sentence in sentences {
            let sentence = sentence.borrow();
            let input = sentence.pieces.view();
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            let token_offsets = sentence
                .token_offsets
                .iter()
                .map(|&offset| offset as i32)
                .collect::<Array1<i32>>();

            let token_lens: Array1<i32> =
                Array1::from_shape_fn((sentence.token_offsets.len(),), |idx| {
                    if idx + 1 < sentence.token_offsets.len() {
                        sentence.token_offsets[idx + 1] as i32 - sentence.token_offsets[idx] as i32
                    } else {
                        sentence.pieces.len() as i32 - sentence.token_offsets[idx] as i32
                    }
                });

            builder.add_without_labels(
                input.view(),
                token_offsets.view(),
                token_lens.view(),
                token_mask.view(),
            );
        }

        builder.into()
    }

    /// Decode biaffine score matrices.
    fn decode_biaffine<S>(
        &self,
        decoder: &ImmutableDependencyEncoder,
        sentences: &mut [S],
        biaffine_score_logits: BiaffineScoreLogits,
    ) -> Result<(), SyntaxDotError>
    where
        S: BorrowMut<SentenceWithPieces>,
    {
        let head_score_logits: ArrayD<f32> =
            (&biaffine_score_logits.head_score_logits).try_into()?;

        // For dependency relations, we only care about the best-scoring relations.
        // This changes the shape from [batch_size, seq_len, seq_len, n_relations] to
        // [batch_size, seq_len, seq_len].
        let best_relations = biaffine_score_logits
            .relation_score_logits
            .argmax(-1, false);
        let best_relations: ArrayD<i32> = (&best_relations).try_into()?;

        for (idx, sentence) in sentences.iter_mut().enumerate() {
            let sentence = sentence.borrow_mut();

            let sent_head_scores = head_score_logits
                .index_axis(Axis(0), idx)
                .slice(s![
                    ..sentence.token_offsets.len() + 1,
                    ..sentence.token_offsets.len() + 1
                ])
                .to_owned();

            let sent_best_relations = best_relations
                .index_axis(Axis(0), idx)
                .slice(s![
                    ..sentence.token_offsets.len() + 1,
                    ..sentence.token_offsets.len() + 1
                ])
                .to_owned();

            decoder.decode(
                sent_head_scores.view().into_dimensionality()?,
                sent_best_relations.view().into_dimensionality()?,
                &mut sentence.sentence,
            );
        }

        Ok(())
    }

    /// Decode sequence labels.
    fn decode_sequence_labels<S>(
        &self,
        sentences: &mut [S],
        sequences_top_k: HashMap<String, TopK>,
    ) -> Result<(), SyntaxDotError>
    where
        S: BorrowMut<SentenceWithPieces>,
    {
        // For each encoder, we get a tensor of shape [batch_size, seq_len, k].
        // Convert the tensors to ndarray tensors, since they are easier to work with
        // in Rust.
        let mut top_k_tensors = HashMap::new();
        for (encoder_name, top_k) in sequences_top_k {
            let top_k_labels: ArrayD<i32> = (&top_k.labels).try_into()?;
            let top_k_probs: ArrayD<f32> = (&top_k.probs).try_into()?;

            top_k_tensors.insert(encoder_name, (top_k_labels, top_k_probs));
        }

        // Extract tensors per sentence.
        for (idx, sentence) in sentences.iter_mut().enumerate() {
            let sentence = sentence.borrow_mut();

            for encoder in self.encoders.iter() {
                let (top_k_labels, top_k_probs) = &top_k_tensors[encoder.name()];

                // Get the sentence and within the sentence the sequence elements
                // that represent tokens.
                let sent_top_k_labels = top_k_labels
                    .index_axis(Axis(0), idx)
                    .slice(s![..sentence.token_offsets.len(), ..])
                    .to_owned();
                let sent_top_k_probs = &top_k_probs
                    .index_axis(Axis(0), idx)
                    .slice(s![..sentence.token_offsets.len(), ..])
                    .to_owned();

                // Collect sentence top-k
                let label_probs: Vec<Vec<EncodingProb<usize>>> = sent_top_k_labels
                    .outer_iter()
                    .zip(sent_top_k_probs.outer_iter())
                    .map(|(token_top_k_labels, token_top_k_probs)| {
                        // Collect token top-k.
                        token_top_k_labels
                            .iter()
                            .zip(token_top_k_probs)
                            .map(|(label, prob)| EncodingProb::new(*label as usize, *prob))
                            .collect()
                    })
                    .collect();

                encoder
                    .encoder()
                    .decode(&label_probs, &mut sentence.sentence)?;
            }
        }

        Ok(())
    }
}
