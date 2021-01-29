use crate::TransformerError;
use tch::{Reduction, Tensor};

trait Reduce {
    type Error;

    fn reduce(&self, t: &Tensor) -> Result<Tensor, Self::Error>;
}

impl Reduce for Reduction {
    type Error = TransformerError;

    fn reduce(&self, t: &Tensor) -> Result<Tensor, Self::Error> {
        match self {
            Reduction::None => Ok(t.shallow_clone()),
            Reduction::Mean => Ok(t.f_mean(t.kind())?),
            Reduction::Sum => Ok(t.f_sum(t.kind())?),
            Reduction::Other(_) => unimplemented!(),
        }
    }
}

/// Cross-entropy loss function.
pub struct CrossEntropyLoss {
    ignore_index: i64,
    label_smoothing: Option<f64>,
    reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Construct the cross-entropy loss function.
    ///
    /// Do not include targets that have `ignore_index` as their value in the
    /// loss computation. If `label_smoothing` is set to *p*, then the correct
    /// label gets probability *1-p* and the probability *p* is distributed
    /// across incorrect labels. `reduction` specifies how the losses should
    /// be reduced/summarized.
    pub fn new(ignore_index: i64, label_smoothing: Option<f64>, reduction: Reduction) -> Self {
        CrossEntropyLoss {
            ignore_index,
            label_smoothing,
            reduction,
        }
    }

    /// Compute the cross-entropy loss.
    ///
    /// `logits` should be the unnormalized probablilities of shape
    /// `[batch_size, seq_len]` and `targets` the gold-standard labels
    /// with shape `[batch_size]`.
    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor, TransformerError> {
        let (_, n_classes) = logits.size2()?;
        let log_probs = logits.f_log_softmax(-1, logits.kind())?;

        match self.label_smoothing {
            Some(label_smoothing) => {
                let token_mask = targets.f_ne(self.ignore_index)?;

                // Do not attempt to use negative indices for the correct target.
                let targets_non_negative =
                    targets.f_where3(&targets.f_ne(self.ignore_index)?, 0)?;

                // Set all labels to label_smoothing and the target to 1-label_smoothing.
                let smoothed_targets = tch::no_grad(|| {
                    Tensor::f_full_like(&log_probs, label_smoothing / (n_classes - 1) as f64)?
                        .f_scatter1(
                            1,
                            &targets_non_negative.f_unsqueeze(1)?,
                            1. - label_smoothing,
                        )
                })?;
                let losses = (smoothed_targets.f_neg()?.f_mul(&log_probs)?).f_sum1(
                    &[-1],
                    false,
                    log_probs.kind(),
                )?;

                Ok(self.reduction.reduce(&losses.masked_select(&token_mask))?)
            }
            None => Ok(log_probs.f_nll_loss::<&Tensor>(
                &targets,
                None,
                self.reduction,
                self.ignore_index,
            )?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryInto;

    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayD};
    use tch::{Reduction, Tensor};

    use crate::loss::CrossEntropyLoss;

    #[test]
    fn cross_entropy_loss_without_label_smoothing() {
        let logits = Tensor::of_slice(&[-1., -1., 1., -1., -1.]).view([1, 5]);
        let targets = Tensor::of_slice(&[2i64]).view([1]);
        let cross_entropy_loss = CrossEntropyLoss::new(-1, None, Reduction::None);
        let loss: ArrayD<f32> = (&cross_entropy_loss.forward(&logits, &targets).unwrap())
            .try_into()
            .unwrap();

        assert_abs_diff_eq!(loss, array![0.432653].into_dyn(), epsilon = 1e-6);
    }

    #[test]
    fn cross_entropy_with_label_smoothing() {
        let logits = Tensor::of_slice(&[-1., -1., 1., -1., -1.]).view([1, 5]);
        let targets = Tensor::of_slice(&[2i64]).view([1]);
        let cross_entropy_loss = CrossEntropyLoss::new(-1, Some(0.1), Reduction::None);
        let loss: ArrayD<f32> = (&cross_entropy_loss.forward(&logits, &targets).unwrap())
            .try_into()
            .unwrap();
        assert_abs_diff_eq!(loss, array![0.632653].into_dyn(), epsilon = 1e-6);
    }
}
