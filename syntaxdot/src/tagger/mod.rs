use std::borrow::{Borrow, BorrowMut};
use std::collections::HashMap;
use std::convert::TryInto;

use ndarray::{Array1, ArrayD, Axis};
use syntaxdot_encoders::{EncodingProb, SentenceDecoder};
use tch::Device;

use crate::encoders::Encoders;
use crate::error::SyntaxDotError;
use crate::input::SentenceWithPieces;
use crate::model::bert::BertModel;
use crate::tensor::{NoLabels, TensorBuilder, Tensors};
use crate::util::seq_len_to_mask;

/// A sequence tagger.
pub struct Tagger {
    device: Device,
    encoders: Encoders,
    model: BertModel,
}

impl Tagger {
    /// Construct a new tagger.
    pub fn new(device: Device, model: BertModel, encoders: Encoders) -> Self {
        Tagger {
            device,
            model,
            encoders,
        }
    }

    /// Tag sentences.
    pub fn tag_sentences(
        &self,
        sentences: &mut [impl BorrowMut<SentenceWithPieces>],
    ) -> Result<(), SyntaxDotError> {
        let top_k_numeric = self.top_k_numeric_(sentences)?;

        for (top_k, sentence) in top_k_numeric.into_iter().zip(sentences.iter_mut()) {
            let sentence = sentence.borrow_mut();

            for encoder in &*self.encoders {
                let encoder_top_k = &top_k[encoder.name()];
                encoder
                    .encoder()
                    .decode(&encoder_top_k, &mut sentence.sentence)?;
            }
        }

        Ok(())
    }

    /// Construct the tensor representations of a batch of sentences.
    fn prepare_batch(
        &self,
        sentences: &[impl Borrow<SentenceWithPieces>],
    ) -> Result<Tensors, SyntaxDotError> {
        let max_seq_len = sentences
            .iter()
            .map(|sentence| sentence.borrow().pieces.len())
            .max()
            .unwrap_or(0);

        let mut builder: TensorBuilder<NoLabels> = TensorBuilder::new(
            sentences.len(),
            max_seq_len,
            self.encoders.iter().map(|encoder| encoder.name()),
        );

        for sentence in sentences {
            let sentence = sentence.borrow();
            let input = sentence.pieces.view();
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            builder.add_without_labels(input.view(), token_mask.view());
        }

        Ok(builder.into())
    }

    /// Get the top-k numeric labels for the sequences.
    #[allow(clippy::type_complexity)]
    fn top_k_numeric_<'a, S>(
        &self,
        sentences: &'a [S],
    ) -> Result<Vec<HashMap<String, Vec<Vec<EncodingProb<usize>>>>>, SyntaxDotError>
    where
        S: Borrow<SentenceWithPieces>,
    {
        let tensors = self.prepare_batch(sentences)?;

        // Convert the top-k labels and arrays into ndarray tensors.
        let mut top_k_tensors = HashMap::new();

        // Get the top-k labels. For each encoder, we get a tensor
        // of shape [batch_size, seq_len, k]. Convert the tensors
        // to ndarray tensors, since they are easier to work with
        // in Rust.
        let mask = seq_len_to_mask(&tensors.seq_lens, tensors.inputs.size()[1]);
        for (encoder_name, top_k) in self.model.top_k(
            &tensors.inputs.to_device(self.device),
            &mask.to_device(self.device),
        ) {
            let (top_k_probs, top_k_labels) = top_k;
            let top_k_labels: ArrayD<i32> = (&top_k_labels).try_into()?;
            let top_k_probs: ArrayD<f32> = (&top_k_probs).try_into()?;

            top_k_tensors.insert(encoder_name, (top_k_labels, top_k_probs));
        }

        // Extract tensors per sentence.
        let mut labels = Vec::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let mut sent_labels = HashMap::new();
            let token_offsets = &sentence.borrow().token_offsets;

            for (encoder_name, (top_k_labels, top_k_probs)) in &top_k_tensors {
                let sent_top_k_labels = top_k_labels
                    .index_axis(Axis(0), idx)
                    .select(Axis(0), &token_offsets);
                let sent_top_k_probs = &top_k_probs
                    .index_axis(Axis(0), idx)
                    .select(Axis(0), &token_offsets);

                // Collect sentence top-k
                let label_probs = sent_top_k_labels
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

                sent_labels.insert(encoder_name.clone(), label_probs);
            }

            labels.push(sent_labels);
        }

        Ok(labels)
    }
}
