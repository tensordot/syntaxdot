use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Deref, DerefMut};

use ndarray::{s, Array1, Array2, ArrayView1};
use syntaxdot_tch_ext::tensor::SumDim;
use syntaxdot_transformers::TransformerError;
use tch::{Device, Kind, Tensor};

use crate::error::SyntaxDotError;

/// Tensors for biaffine encodings.
#[derive(Debug, PartialEq)]
pub struct BiaffineTensors<T> {
    pub heads: T,
    pub relations: T,
}

impl BiaffineTensors<Array2<i64>> {
    fn from_shape(batch_size: usize, time_steps: usize) -> Self {
        BiaffineTensors {
            heads: Array2::from_elem((batch_size, time_steps), -1),
            relations: Array2::from_elem((batch_size, time_steps), -1),
        }
    }
}

impl BiaffineTensors<Tensor> {
    pub fn to_device(&self, device: Device) -> Self {
        BiaffineTensors {
            heads: self.heads.to_device(device),
            relations: self.relations.to_device(device),
        }
    }
}

/// Labels per encoder.
pub struct LabelTensor {
    inner: HashMap<String, Array2<i64>>,
}

impl LabelTensor {
    fn from_shape(
        encoder_names: impl IntoIterator<Item = impl Into<String>>,
        batch_size: usize,
        time_steps: usize,
    ) -> Self {
        let labels = encoder_names
            .into_iter()
            .map(Into::into)
            .map(|encoder_name| (encoder_name, Array2::zeros((batch_size, time_steps))))
            .collect();

        LabelTensor { inner: labels }
    }
}

impl Deref for LabelTensor {
    type Target = HashMap<String, Array2<i64>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for LabelTensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Build Torch `Tensor`s from `ndarray` vectors.
pub struct TensorBuilder {
    biaffine_encodings: Option<BiaffineTensors<Array2<i64>>>,
    current_sequence: usize,
    inputs: Array2<i64>,
    labels: Option<LabelTensor>,
    token_offsets: Array2<i32>,
    token_len: Array2<i32>,
    token_mask: Array2<i32>,
    seq_lens: Array1<i32>,
}

impl TensorBuilder {
    /// Create a new `TensorBuilder` without labels.
    ///
    /// Creates a new builder with the given batch size and number of
    /// time steps.
    pub fn new_without_labels(
        batch_size: usize,
        max_seq_len: usize,
        max_tokens_len: usize,
    ) -> Self {
        TensorBuilder {
            biaffine_encodings: None,
            current_sequence: 0,
            inputs: Array2::zeros((batch_size, max_seq_len)),
            token_offsets: Array2::from_elem((batch_size, max_tokens_len), -1),
            token_len: Array2::from_elem((batch_size, max_tokens_len), -1),
            token_mask: Array2::zeros((batch_size, max_seq_len)),
            labels: None,
            seq_lens: Array1::zeros((batch_size,)),
        }
    }

    /// Create a new `TensorBuilder` with labels.
    ///
    /// Creates a new builder with the given batch size, number of time steps,
    /// and encoder names.
    pub fn new_with_labels(
        batch_size: usize,
        max_seq_len: usize,
        max_tokens_len: usize,
        biaffine_encoder: bool,
        encoder_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let biaffine_encodings = if biaffine_encoder {
            Some(BiaffineTensors::from_shape(batch_size, max_tokens_len))
        } else {
            None
        };

        TensorBuilder {
            biaffine_encodings,
            current_sequence: 0,
            inputs: Array2::zeros((batch_size, max_seq_len)),
            token_offsets: Array2::from_elem((batch_size, max_tokens_len), -1),
            token_len: Array2::from_elem((batch_size, max_tokens_len), -1),
            token_mask: Array2::zeros((batch_size, max_seq_len)),
            labels: Some(LabelTensor::from_shape(
                encoder_names,
                batch_size,
                max_tokens_len,
            )),
            seq_lens: Array1::zeros((batch_size,)),
        }
    }
}

impl TensorBuilder {
    /// Add an instance without labels.
    ///
    /// The `token_mask` should be a mask which is set to `true` for
    /// inputs that correspond to the initial word piece of a token.
    pub fn add_without_labels(
        &mut self,
        input: ArrayView1<i64>,
        token_indices: ArrayView1<i32>,
        token_lens: ArrayView1<i32>,
        token_mask: ArrayView1<i32>,
    ) {
        assert!(
            self.current_sequence < self.inputs.shape()[0],
            "TensorBuilder is already filled."
        );

        #[allow(clippy::deref_addrof)]
        self.inputs
            .row_mut(self.current_sequence)
            .slice_mut(s![0..input.len()])
            .assign(&input);

        self.token_offsets
            .row_mut(self.current_sequence)
            .slice_mut(s![0..token_indices.len()])
            .assign(&token_indices);

        self.token_len
            .row_mut(self.current_sequence)
            .slice_mut(s![0..token_lens.len()])
            .assign(&token_lens);

        self.token_mask
            .row_mut(self.current_sequence)
            .slice_mut(s![0..token_mask.len()])
            .assign(&token_mask);

        self.seq_lens[self.current_sequence] = input.len() as i32;

        self.current_sequence += 1
    }

    /// Add an instance with labels.
    pub fn add_with_labels(
        &mut self,
        input: ArrayView1<i64>,
        biaffine_labels: Option<(Array1<i64>, Array1<i64>)>,
        sequence_labels: HashMap<&str, Array1<i64>>,
        token_offsets: ArrayView1<i32>,
        token_lens: ArrayView1<i32>,
        token_mask: ArrayView1<i32>,
    ) {
        assert!(
            self.current_sequence < self.inputs.shape()[0],
            "TensorBuilder is already filled."
        );

        assert_eq!(
            self.labels.as_ref().unwrap().len(),
            sequence_labels.len(),
            "Expected labels for {} encoders, got labels for {}",
            self.labels.as_ref().unwrap().len(),
            sequence_labels.len(),
        );

        assert!(
            (self.biaffine_encodings.is_some() == biaffine_labels.is_some()),
            "Expected biaffine encodings, none were provided"
        );

        if let (Some(biaffine_encodings), Some(instance_biaffine_encodings)) =
            (self.biaffine_encodings.as_mut(), biaffine_labels)
        {
            assert_eq!(
                instance_biaffine_encodings.0.len(),
                token_offsets.len(),
                "Biaffine heads has length {}, but the sentence length is {}",
                instance_biaffine_encodings.0.len(),
                token_offsets.len()
            );
            assert_eq!(
                instance_biaffine_encodings.1.len(),
                token_offsets.len(),
                "Biaffine relations has length {}, but the sentence length is {}",
                instance_biaffine_encodings.1.len(),
                token_offsets.len()
            );

            biaffine_encodings
                .heads
                .row_mut(self.current_sequence)
                .slice_mut(s![0..token_offsets.len()])
                .assign(&instance_biaffine_encodings.0);

            biaffine_encodings
                .relations
                .row_mut(self.current_sequence)
                .slice_mut(s![0..token_offsets.len()])
                .assign(&instance_biaffine_encodings.1);
        };

        for (encoder_name, labels) in sequence_labels {
            assert_eq!(
                labels.len(),
                token_offsets.len(),
                "Input for encoder {} has length {}, but the offsets length is {}",
                encoder_name,
                labels.len(),
                token_offsets.len()
            );

            #[allow(clippy::deref_addrof)]
            self.labels
                .as_mut()
                .unwrap()
                .get_mut(encoder_name)
                .unwrap_or_else(|| panic!("Undefined encoder: {}", encoder_name))
                .row_mut(self.current_sequence)
                .slice_mut(s![0..labels.len()])
                .assign(&labels)
        }

        self.add_without_labels(input, token_offsets, token_lens, token_mask);
    }
}

/// Tensors constructed by `TensorBuilder`.
#[derive(Debug)]
pub struct Tensors {
    /// Input representations.
    pub inputs: Tensor,

    /// Biaffine encodings.
    pub biaffine_encodings: Option<BiaffineTensors<Tensor>>,

    /// Labels.
    pub labels: Option<HashMap<String, Tensor>>,

    /// Sequence lengths.
    pub seq_lens: SequenceLengths,

    /// Token offsets.
    pub token_spans: TokenSpans,
}

impl From<TensorBuilder> for Tensors {
    fn from(builder: TensorBuilder) -> Self {
        let labels = builder.labels.map(|labels| {
            labels
                .inner
                .into_iter()
                .map(|(encoder_name, matrix)| (encoder_name, matrix.try_into().unwrap()))
                .collect()
        });

        let biaffine_encodings = builder.biaffine_encodings.map(|encodings| BiaffineTensors {
            heads: encodings.heads.try_into().unwrap(),
            relations: encodings.relations.try_into().unwrap(),
        });

        Tensors {
            inputs: builder.inputs.try_into().unwrap(),
            biaffine_encodings,
            labels,
            seq_lens: SequenceLengths::new(builder.seq_lens.try_into().unwrap()),
            token_spans: TokenSpans::new(
                Tensor::try_from(builder.token_offsets)
                    .unwrap()
                    .to_kind(Kind::Int64),
                Tensor::try_from(builder.token_len)
                    .unwrap()
                    .to_kind(Kind::Int64),
            ),
        }
    }
}

/// Sequence word/sentence piece lengths.
#[derive(Debug)]
pub struct SequenceLengths {
    inner: Tensor,
}

impl SequenceLengths {
    fn new(seq_lens: Tensor) -> Self {
        Self { inner: seq_lens }
    }

    /// Convert sequence lengths to masks.
    pub fn attention_mask(&self) -> Result<Tensor, SyntaxDotError> {
        let max_len = i64::try_from(self.inner.max())?;
        let batch_size = self.inner.size()[0];
        Ok(Tensor::f_arange(max_len, (Kind::Int, self.inner.device()))?
            // Construct a matrix [batch_size, max_len] where each row
            // is 0..(max_len - 1).
            .f_repeat([batch_size])?
            .f_view_([batch_size, max_len])?
            // Time steps less than the length in the sequence lengths are active.
            .f_lt_tensor(&self.inner.unsqueeze(1))?
            // For some reason the kind is Int?
            .to_kind(Kind::Bool))
    }
}

impl Deref for SequenceLengths {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Token spans
#[derive(Debug)]
pub struct TokenSpans {
    offsets: Tensor,
    lens: Tensor,
}

impl TokenSpans {
    pub(crate) fn new(token_offsets: Tensor, token_lens: Tensor) -> Self {
        Self {
            offsets: token_offsets,
            lens: token_lens,
        }
    }

    /// Copy the token offsets to the given device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            offsets: self.offsets.to_device(device),
            lens: self.lens.to_device(device),
        }
    }

    /// Get the token lengths.
    pub fn lens(&self) -> &Tensor {
        &self.lens
    }

    /// Get the token offsets.
    pub fn offsets(&self) -> &Tensor {
        &self.offsets
    }

    /// Create a token mask from token offsets.
    pub fn token_mask(&self) -> Result<TokenMask, SyntaxDotError> {
        Ok(TokenMask {
            inner: self.offsets.f_ne(-1)?,
        })
    }

    /// Get the sequence lengths of the sequences in the batch.
    pub fn seq_lens(&self) -> Result<Tensor, SyntaxDotError> {
        Ok(self
            .token_mask()?
            .f_sum_dim(-1, false, self.offsets().kind())?)
    }

    /// Get the token spans with the ROOT depedency token prepended.
    pub fn with_root(&self) -> Result<TokenSpansWithRoot, TransformerError> {
        let (batch_size, _) = self.offsets.size2()?;

        let root_offset = Tensor::from(0)
            .f_view([1, 1])?
            .f_expand([batch_size, 1], true)?
            .to_device(self.offsets.device());
        let offsets = Tensor::f_cat(&[&root_offset, &self.offsets], 1)?;

        let root_len = Tensor::from(1)
            .f_view([1, 1])?
            .f_expand([batch_size, 1], true)?
            .to_device(self.lens.device());
        let lens = Tensor::f_cat(&[&root_len, &self.lens], 1)?;

        Ok(TokenSpansWithRoot::new(offsets, lens))
    }
}

/// Token spans
#[derive(Debug)]
pub struct TokenSpansWithRoot {
    offsets: Tensor,
    lens: Tensor,
}

impl TokenSpansWithRoot {
    pub(crate) fn new(offsets: Tensor, lens: Tensor) -> Self {
        Self { offsets, lens }
    }

    /// Get the token lengths.
    pub fn lens(&self) -> &Tensor {
        &self.lens
    }

    /// Get the token offsets.
    pub fn offsets(&self) -> &Tensor {
        &self.offsets
    }
}

impl TokenSpansWithRoot {
    /// Create a token mask from token offsets.
    pub fn token_mask(&self) -> Result<TokenMask, TransformerError> {
        Ok(TokenMask {
            inner: self.offsets.f_ne(-1)?,
        })
    }
}

/// Token mask.
#[derive(Debug)]
pub struct TokenMask {
    inner: Tensor,
}

impl TokenMask {
    pub fn with_root(&self) -> Result<TokenMaskWithRoot, SyntaxDotError> {
        let (batch_size, _seq_len) = self.inner.size2()?;

        let root_mask = Tensor::from(true)
            .f_expand([batch_size, 1], true)?
            .to_device(self.inner.device());

        let token_mask_with_root = Tensor::f_cat(&[&root_mask, &self.inner], -1)?;

        Ok(TokenMaskWithRoot::new(token_mask_with_root))
    }
}

impl Deref for TokenMask {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Token mask with prepended mask for a dependency root.
#[derive(Debug)]
pub struct TokenMaskWithRoot {
    inner: Tensor,
}

impl TokenMaskWithRoot {
    fn new(token_mask_with_root: Tensor) -> Self {
        Self {
            inner: token_mask_with_root,
        }
    }
}

impl Deref for TokenMaskWithRoot {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use tch::Tensor;

    use super::{TensorBuilder, Tensors};
    use crate::tensor::{BiaffineTensors, SequenceLengths, TokenSpans};

    #[test]
    fn attention_masking_is_correct() {
        let seq_lens = SequenceLengths::new(Tensor::from_slice(&[3, 5, 1]));
        assert_eq!(
            seq_lens.attention_mask().unwrap(),
            Tensor::from_slice(&[
                true, true, true, false, false, // Sequence 0
                true, true, true, true, true, // Sequence 1
                true, false, false, false, false, // Sequence 2
            ])
            .view([3, 5])
        );
    }

    #[test]
    fn instances_are_added() {
        let mut builder: TensorBuilder = TensorBuilder::new_without_labels(2, 3, 2);
        builder.add_without_labels(
            arr1(&[1, 2]).view(),
            arr1(&[0]).view(),
            arr1(&[1]).view(),
            arr1(&[1, 0]).view(),
        );
        builder.add_without_labels(
            arr1(&[3, 4, 5]).view(),
            arr1(&[0, 2]).view(),
            arr1(&[2, 1]).view(),
            arr1(&[1, 0, 1]).view(),
        );

        let tensors: Tensors = builder.into();

        // No labels.
        assert_eq!(tensors.labels, None);

        assert_eq!(*tensors.seq_lens, Tensor::from_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::from_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
        );
    }

    #[test]
    fn instances_are_added_with_labels() {
        let mut builder: TensorBuilder =
            TensorBuilder::new_with_labels(2, 3, 2, true, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            Some((arr1(&[1]), arr1(&[2]))),
            vec![("a", arr1(&[12])), ("b", arr1(&[21]))]
                .into_iter()
                .collect(),
            arr1(&[0]).view(),
            arr1(&[1]).view(),
            arr1(&[1, 0]).view(),
        );
        builder.add_with_labels(
            arr1(&[3, 4, 5]).view(),
            Some((arr1(&[0, 1]), arr1(&[3, 1]))),
            vec![("a", arr1(&[13, 15])), ("b", arr1(&[24, 25]))]
                .into_iter()
                .collect(),
            arr1(&[0, 2]).view(),
            arr1(&[2, 1]).view(),
            arr1(&[1, 0, 1]).view(),
        );

        let tensors: Tensors = builder.into();

        // Biaffine encodings
        assert_eq!(
            tensors.biaffine_encodings,
            Some(BiaffineTensors {
                heads: Tensor::from_slice(&[1, -1, 0, 1]).reshape(&[2, 2]),
                relations: Tensor::from_slice(&[2, -1, 3, 1]).reshape(&[2, 2])
            })
        );

        // Labels.
        assert_eq!(
            tensors.labels,
            Some(
                vec![
                    (
                        "a".to_string(),
                        Tensor::from_slice(&[12, 0, 13, 15]).reshape(&[2, 2])
                    ),
                    (
                        "b".to_string(),
                        Tensor::from_slice(&[21, 0, 24, 25]).reshape(&[2, 2])
                    )
                ]
                .into_iter()
                .collect()
            )
        );

        assert_eq!(*tensors.seq_lens, Tensor::from_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::from_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
        );
    }

    #[should_panic]
    #[test]
    fn panics_when_labels_and_mask_len_differ() {
        let mut builder: TensorBuilder =
            TensorBuilder::new_with_labels(2, 3, 1, false, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            None,
            vec![("a", arr1(&[11])), ("b", arr1(&[21, 22]))]
                .into_iter()
                .collect(),
            arr1(&[0]).view(),
            arr1(&[1]).view(),
            arr1(&[1, 0]).view(),
        );
    }

    #[should_panic]
    #[test]
    fn panics_when_too_many_instances_pushed() {
        let mut builder: TensorBuilder = TensorBuilder::new_without_labels(1, 3, 2);
        builder.add_without_labels(
            arr1(&[1, 2]).view(),
            arr1(&[0]).view(),
            arr1(&[1]).view(),
            arr1(&[1, 0]).view(),
        );
        builder.add_without_labels(
            arr1(&[3, 4, 5]).view(),
            arr1(&[0, 2]).view(),
            arr1(&[2, 1]).view(),
            arr1(&[1, 0, 1]).view(),
        );
    }

    #[should_panic]
    #[test]
    fn panics_when_labels_for_encoder_missing() {
        let mut builder: TensorBuilder =
            TensorBuilder::new_with_labels(2, 3, 1, false, vec!["a", "b"]);
        builder.add_with_labels(
            arr1(&[1, 2]).view(),
            None,
            vec![("b", arr1(&[21, 22]))].into_iter().collect(),
            arr1(&[0]).view(),
            arr1(&[1]).view(),
            arr1(&[1, 0]).view(),
        );
    }

    #[test]
    fn token_masking_is_correct() {
        let token_offsets = TokenSpans::new(
            Tensor::from_slice2(&[&[1, 3, 5, -1, -1], &[1, 2, 8, 11, 13]]),
            Tensor::from_slice2(&[&[2, 2, 1, -1, -1], &[1, 6, 3, 2, 1]]),
        );
        assert_eq!(
            *token_offsets.token_mask().unwrap(),
            Tensor::from_slice(&[
                true, true, true, false, false, // Sequence 0
                true, true, true, true, true // Sequence 1
            ])
            .view([2, 5])
        );
    }

    #[test]
    fn token_masking_with_root_is_correct() {
        let token_offsets = TokenSpans::new(
            Tensor::from_slice2(&[&[1, 3, 5, -1, -1], &[1, 2, 8, 11, 13]]),
            Tensor::from_slice2(&[&[2, 2, 1, -1, -1], &[1, 6, 3, 2, 1]]),
        );

        assert_eq!(
            *token_offsets.token_mask().unwrap().with_root().unwrap(),
            Tensor::from_slice(&[
                true, true, true, true, false, false, // Sequence 0
                true, true, true, true, true, true // Sequence 1
            ])
            .view([2, 6])
        );
    }

    #[test]
    fn token_sequence_lengths_are_correct() {
        let token_offsets = TokenSpans::new(
            Tensor::from_slice2(&[&[1, 3, 5, -1, -1], &[1, 2, 8, 11, 13]]),
            Tensor::from_slice2(&[&[2, 2, 1, -1, -1], &[1, 6, 3, 2, 1]]),
        );
        assert_eq!(
            token_offsets.seq_lens().unwrap(),
            Tensor::from_slice(&[3, 5])
        );
    }
}
