use std::collections::HashMap;

use ndarray::Array1;
use syntaxdot_encoders::SentenceEncoder;

use crate::encoders::NamedEncoder;
use crate::error::SyntaxDotError;
use crate::input::SentenceWithPieces;
use crate::tensor::{TensorBuilder, Tensors};

/// An iterator returning input and (optionally) output tensors.
pub struct TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, conllu::IOError>>,
{
    pub batch_size: usize,
    pub encoders: Option<&'a [NamedEncoder]>,
    pub sentences: I,
}

impl<'a, I> TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, conllu::IOError>>,
{
    fn next_with_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces>,
        max_seq_len: usize,
        encoders: &'a [NamedEncoder],
    ) -> Option<Result<Tensors, SyntaxDotError>> {
        let mut builder = TensorBuilder::new_with_labels(
            tokenized_sentences.len(),
            max_seq_len,
            encoders.iter().map(NamedEncoder::name),
        );

        for sentence in tokenized_sentences {
            let input = sentence.pieces;
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            let mut encoder_labels = HashMap::with_capacity(encoders.len());
            for encoder in encoders {
                let encoding = match encoder.encoder().encode(&sentence.sentence) {
                    Ok(encoding) => encoding,
                    Err(err) => return Some(Err(err.into())),
                };

                let mut labels = Array1::from_elem((input.len(),), 1i64);
                for (encoding, offset) in encoding.into_iter().zip(&sentence.token_offsets) {
                    labels[*offset] = encoding as i64;
                }

                encoder_labels.insert(encoder.name(), labels);
            }

            builder.add_with_labels(input.view(), encoder_labels, token_mask.view());
        }

        Some(Ok(builder.into()))
    }

    fn next_without_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces>,
        max_seq_len: usize,
    ) -> Option<Result<Tensors, SyntaxDotError>> {
        let mut builder: TensorBuilder =
            TensorBuilder::new_without_labels(tokenized_sentences.len(), max_seq_len);

        for sentence in tokenized_sentences {
            let input = sentence.pieces;
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            builder.add_without_labels(input.view(), token_mask.view());
        }

        Some(Ok(builder.into()))
    }
}

impl<'a, I> Iterator for TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, conllu::IOError>>,
{
    type Item = Result<Tensors, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_sentences = Vec::with_capacity(self.batch_size);
        while let Some(sentence) = self.sentences.next() {
            let sentence = match sentence {
                Ok(sentence) => sentence,
                Err(err) => return Some(Err(err.into())),
            };
            batch_sentences.push(sentence);
            if batch_sentences.len() == self.batch_size {
                break;
            }
        }

        // Check whether the reader is exhausted.
        if batch_sentences.is_empty() {
            return None;
        }

        let max_seq_len = batch_sentences
            .iter()
            .map(|s| s.pieces.len())
            .max()
            .unwrap_or(0);

        match self.encoders {
            Some(encoders) => self.next_with_labels(batch_sentences, max_seq_len, encoders),
            None => self.next_without_labels(batch_sentences, max_seq_len),
        }
    }
}
