use std::collections::VecDeque;
use std::fs;

use anyhow::{Context, Result};
use tch::nn::VarStore;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum CompletedUnit<P> {
    /// A batch is completed with the given performance.
    Batch(P),

    /// An epoch is completed with the given performance.
    ///
    /// The performance is of an epoch is typically evaluated against
    /// a validation set.
    Epoch(P),
}

/// Trait for model savers.
pub trait Save<P> {
    /// Save a model
    ///
    /// Calling this method amounts to a request to save a
    /// model. Whether an actual model is saved depends on the
    /// implementor. E.g. `EpochSaver` only saves a model for
    /// each epoch, so requests to save at a completed batch
    /// are ignored.
    ///
    /// The performance should be that a better performance compares
    /// as larger. If smaller is better in a performance measure, the
    /// actual measure can be wrapped in `std::cmp::Reverse` to
    /// reverse the ordering.
    fn save(&mut self, vs: &VarStore, completed: CompletedUnit<P>) -> Result<()>;
}

/// Save best epochs with the best performance so far.
#[derive(Clone)]
pub struct BestEpochSaver<P> {
    best_epoch_performance: Option<P>,
    best_epoch_paths: Option<VecDeque<String>>,
    epoch: usize,
    keep_best_epochs: Option<usize>,
    prefix: String,
}

impl<P> BestEpochSaver<P> {
    pub fn new(prefix: impl Into<String>, keep_best_epochs: Option<usize>) -> Self {
        BestEpochSaver {
            best_epoch_performance: None,
            best_epoch_paths: keep_best_epochs.map(VecDeque::with_capacity),
            epoch: 0,
            keep_best_epochs,
            prefix: prefix.into(),
        }
    }

    fn cleanup_old_best_steps(&mut self, step_path: String) {
        if let Some(best_epoch_paths) = &mut self.best_epoch_paths {
            eprintln!(
                "best len: {}, best cap: {}",
                best_epoch_paths.len(),
                best_epoch_paths.capacity()
            );
            if best_epoch_paths.len() == self.keep_best_epochs.unwrap() {
                let cleanup_step = best_epoch_paths.pop_front().expect("No steps?");
                if let Err(err) = fs::remove_file(&cleanup_step) {
                    log::error!("Cannot remove step parameters {}: {}", cleanup_step, err);
                }
            }

            best_epoch_paths.push_back(step_path);
        }
    }
}

impl<P> Save<P> for BestEpochSaver<P>
where
    P: PartialOrd,
{
    fn save(&mut self, vs: &VarStore, completed: CompletedUnit<P>) -> Result<()> {
        if let CompletedUnit::Epoch(perf) = completed {
            let improvement = match self.best_epoch_performance {
                Some(ref mut best) => {
                    if perf > *best {
                        *best = perf;
                        true
                    } else {
                        false
                    }
                }
                None => {
                    self.best_epoch_performance = Some(perf);
                    true
                }
            };

            if improvement {
                let path = format!("{}epoch-{}", self.prefix, self.epoch);
                vs.save(&path).context(format!(
                    "Cannot save variable store for epoch {}",
                    self.epoch
                ))?;

                self.cleanup_old_best_steps(path)
            }

            self.epoch += 1;
        }

        Ok(())
    }
}
