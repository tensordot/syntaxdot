use std::collections::HashMap;

use ndarray::Array1;
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_encoders::SentenceEncoder;
use syntaxdot_tokenizers::SentenceWithPieces;

use crate::encoders::NamedEncoder;
use crate::error::SyntaxDotError;
use crate::tensor::{TensorBuilder, Tensors};

pub trait BatchedTensors<'a> {
    /// Get an iterator over batch tensors.
    ///
    /// The sequence labels using the `encoders`, syntactic
    /// dependencies using `biaffine_encoder`.
    ///
    /// If `encoders` is not `None`, output tensors will be created
    /// for the sequence labels in the data set.
    ///
    /// If `biaffine_encoder` is not `None`, output tensors will be
    /// created dependency heads and relations.
    #[allow(clippy::type_complexity)]
    fn batched_tensors(
        self,
        biaffine_encoder: Option<&'a ImmutableDependencyEncoder>,
        encoders: Option<&'a [NamedEncoder]>,
        batch_size: usize,
    ) -> TensorIter<'a, Box<dyn Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>> + 'a>>;
}

impl<'a, I> BatchedTensors<'a> for I
where
    I: 'a + Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    #[allow(clippy::type_complexity)]
    fn batched_tensors(
        self,
        biaffine_encoder: Option<&'a ImmutableDependencyEncoder>,
        encoders: Option<&'a [NamedEncoder]>,
        batch_size: usize,
    ) -> TensorIter<'a, Box<dyn Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>> + 'a>>
    {
        TensorIter {
            batch_size,
            biaffine_encoder,
            encoders,
            sentences: Box::new(self),
        }
    }
}

/// An iterator returning input and (optionally) output tensors.
pub struct TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    pub batch_size: usize,
    pub biaffine_encoder: Option<&'a ImmutableDependencyEncoder>,
    pub encoders: Option<&'a [NamedEncoder]>,
    pub sentences: I,
}

impl<'a, I> TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    fn next_with_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces>,
        max_seq_len: usize,
        max_tokens_len: usize,
        biaffine_encoder: Option<&'a ImmutableDependencyEncoder>,
        encoders: &'a [NamedEncoder],
    ) -> Result<Tensors, SyntaxDotError> {
        let mut builder = TensorBuilder::new_with_labels(
            tokenized_sentences.len(),
            max_seq_len,
            max_tokens_len,
            biaffine_encoder.is_some(),
            encoders.iter().map(NamedEncoder::name),
        );

        for sentence in tokenized_sentences {
            let mut token_mask = Array1::zeros((sentence.pieces.len(),));
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

            let biaffine_encoding = match Self::encode_biaffine(biaffine_encoder, &sentence) {
                Ok(biaffine_encoding) => biaffine_encoding,
                Err(err) => return Err(err),
            };

            let sequence_encoding = match Self::encode_sequence(encoders, &sentence) {
                Ok(sequence_encoding) => sequence_encoding,
                Err(err) => return Err(err),
            };

            builder.add_with_labels(
                sentence.pieces.view(),
                biaffine_encoding,
                sequence_encoding,
                token_offsets.view(),
                token_lens.view(),
                token_mask.view(),
            );
        }

        Ok(builder.into())
    }

    fn encode_sequence<'e>(
        encoders: &'e [NamedEncoder],
        sentence: &SentenceWithPieces,
    ) -> Result<HashMap<&'e str, Array1<i64>>, SyntaxDotError> {
        let mut encoder_labels = HashMap::with_capacity(encoders.len());
        for encoder in encoders {
            let encoding = match encoder.encoder().encode(&sentence.sentence) {
                Ok(encoding) => encoding,
                Err(err) => return Err(err.into()),
            };

            let labels = encoding.into_iter().map(|label| label as i64).collect();

            encoder_labels.insert(encoder.name(), labels);
        }
        Ok(encoder_labels)
    }

    #[allow(clippy::type_complexity)]
    fn encode_biaffine(
        biaffine_encoder: Option<&ImmutableDependencyEncoder>,
        sentence: &SentenceWithPieces,
    ) -> Result<Option<(Array1<i64>, Array1<i64>)>, SyntaxDotError> {
        let encoding = match biaffine_encoder {
            Some(biaffine_encoder) => {
                let encoding = biaffine_encoder.encode(&sentence.sentence)?;

                let dependency_heads = encoding.heads.into_iter().map(|head| head as i64).collect();
                let dependency_labels = encoding
                    .relations
                    .into_iter()
                    .map(|relation| relation as i64)
                    .collect();

                Some((dependency_heads, dependency_labels))
            }
            None => None,
        };

        Ok(encoding)
    }

    fn next_without_labels(
        &mut self,
        tokenized_sentences: Vec<SentenceWithPieces>,
        max_seq_len: usize,
        max_tokens_len: usize,
    ) -> Tensors {
        let mut builder: TensorBuilder = TensorBuilder::new_without_labels(
            tokenized_sentences.len(),
            max_seq_len,
            max_tokens_len,
        );

        for sentence in tokenized_sentences {
            let input = sentence.pieces;
            let mut token_mask = Array1::zeros((input.len(),));
            for token_idx in &sentence.token_offsets {
                token_mask[*token_idx] = 1;
            }

            let token_offsets = sentence
                .token_offsets
                .iter()
                .map(|&offset| offset as i32)
                .collect::<Array1<i32>>();

            let token_lens: Array1<i32> = Array1::from_shape_fn((token_offsets.len(),), |idx| {
                if idx + 1 < token_offsets.len() {
                    token_offsets[idx + 1] as i32 - token_offsets[idx] as i32
                } else {
                    input.len() as i32 - token_offsets[idx] as i32
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
}

impl<'a, I> Iterator for TensorIter<'a, I>
where
    I: Iterator<Item = Result<SentenceWithPieces, SyntaxDotError>>,
{
    type Item = Result<Tensors, SyntaxDotError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_sentences = Vec::with_capacity(self.batch_size);
        for sentence in &mut self.sentences {
            let sentence = match sentence {
                Ok(sentence) => sentence,
                Err(err) => return Some(Err(err)),
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

        let max_tokens_len = batch_sentences
            .iter()
            .map(|s| s.token_offsets.len())
            .max()
            .unwrap_or(0);

        Some(match self.encoders {
            Some(encoders) => self.next_with_labels(
                batch_sentences,
                max_seq_len,
                max_tokens_len,
                self.biaffine_encoder,
                encoders,
            ),
            None => Ok(self.next_without_labels(batch_sentences, max_seq_len, max_tokens_len)),
        })
    }
}
