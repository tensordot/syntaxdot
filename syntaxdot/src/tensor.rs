use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::ops::{Deref, DerefMut};

use ndarray::{s, Array1, Array2, ArrayView1};
use tch::{Device, Kind, Tensor};

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

        self.add_without_labels(input, token_offsets, token_mask);
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

    /// Token mask.
    pub token_offsets: Tensor,

    /// Sequence lengths.
    pub seq_lens: Tensor,
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
            token_offsets: Tensor::try_from(builder.token_offsets)
                .unwrap()
                .to_kind(Kind::Int64),
            seq_lens: builder.seq_lens.try_into().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use tch::Tensor;

    use super::{TensorBuilder, Tensors};
    use crate::tensor::BiaffineTensors;

    #[test]
    fn instances_are_added() {
        let mut builder: TensorBuilder = TensorBuilder::new_without_labels(2, 3, 2);
        builder.add_without_labels(
            arr1(&[1, 2]).view(),
            arr1(&[0]).view(),
            arr1(&[1, 0]).view(),
        );
        builder.add_without_labels(
            arr1(&[3, 4, 5]).view(),
            arr1(&[0, 2]).view(),
            arr1(&[1, 0, 1]).view(),
        );

        let tensors: Tensors = builder.into();

        // No labels.
        assert_eq!(tensors.labels, None);

        assert_eq!(tensors.seq_lens, Tensor::of_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::of_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
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
            arr1(&[1, 0]).view(),
        );
        builder.add_with_labels(
            arr1(&[3, 4, 5]).view(),
            Some((arr1(&[0, 1]), arr1(&[3, 1]))),
            vec![("a", arr1(&[13, 15])), ("b", arr1(&[24, 25]))]
                .into_iter()
                .collect(),
            arr1(&[0, 2]).view(),
            arr1(&[1, 0, 1]).view(),
        );

        let tensors: Tensors = builder.into();

        // Biaffine encodings
        assert_eq!(
            tensors.biaffine_encodings,
            Some(BiaffineTensors {
                heads: Tensor::of_slice(&[1, -1, 0, 1]).reshape(&[2, 2]),
                relations: Tensor::of_slice(&[2, -1, 3, 1]).reshape(&[2, 2])
            })
        );

        // Labels.
        assert_eq!(
            tensors.labels,
            Some(
                vec![
                    (
                        "a".to_string(),
                        Tensor::of_slice(&[12, 0, 13, 15]).reshape(&[2, 2])
                    ),
                    (
                        "b".to_string(),
                        Tensor::of_slice(&[21, 0, 24, 25]).reshape(&[2, 2])
                    )
                ]
                .into_iter()
                .collect()
            )
        );

        assert_eq!(tensors.seq_lens, Tensor::of_slice(&[2, 3]));
        assert_eq!(
            tensors.inputs,
            Tensor::of_slice(&[1, 2, 0, 3, 4, 5]).reshape(&[2, 3])
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
            arr1(&[1, 0]).view(),
        );
        builder.add_without_labels(
            arr1(&[3, 4, 5]).view(),
            arr1(&[0, 2]).view(),
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
            arr1(&[1, 0]).view(),
        );
    }
}
