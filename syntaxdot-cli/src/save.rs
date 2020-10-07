use std::marker::PhantomData;

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

/// Save epochs.
#[derive(Clone)]
pub struct EpochSaver<P> {
    epoch: usize,
    prefix: String,
    _phantom: PhantomData<P>,
}

impl<P> EpochSaver<P> {
    pub fn new(prefix: impl Into<String>) -> Self {
        EpochSaver {
            prefix: prefix.into(),
            epoch: 0,
            _phantom: PhantomData,
        }
    }
}

impl<P> Save<P> for EpochSaver<P> {
    fn save(&mut self, vs: &VarStore, completed: CompletedUnit<P>) -> Result<()> {
        if let CompletedUnit::Epoch(_) = completed {
            vs.save(format!("{}epoch-{}", self.prefix, self.epoch))
                .context(format!(
                    "Cannot save variable store for epoch {}",
                    self.epoch
                ))?;
            self.epoch += 1;
        }

        Ok(())
    }
}

/// Save best epochs with the best performance so far.
#[derive(Clone)]
pub struct BestEpochSaver<P> {
    best_epoch_performance: Option<P>,
    epoch: usize,
    prefix: String,
}

impl<P> BestEpochSaver<P> {
    pub fn new(prefix: impl Into<String>) -> Self {
        BestEpochSaver {
            best_epoch_performance: None,
            epoch: 0,
            prefix: prefix.into(),
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
                vs.save(format!("{}epoch-{}", self.prefix, self.epoch))
                    .context(format!(
                        "Cannot save variable store for epoch {}",
                        self.epoch
                    ))?;
            }

            self.epoch += 1;
        }

        Ok(())
    }
}

/// Save every epoch and N batches.
#[derive(Clone)]
pub struct EpochAndBatchesSaver<P> {
    batch: usize,
    epoch: usize,
    epoch_batch: usize,
    n_batches: usize,
    prefix: String,
    _phantom: PhantomData<P>,
}

impl<P> EpochAndBatchesSaver<P> {
    /// Construct a saver that saves every epoch and N batches.
    pub fn new(prefix: impl Into<String>, n_batches: usize) -> Self {
        EpochAndBatchesSaver {
            batch: 0,
            epoch: 0,
            epoch_batch: 0,
            n_batches,
            prefix: prefix.into(),
            _phantom: PhantomData,
        }
    }
}

impl<P> Save<P> for EpochAndBatchesSaver<P> {
    fn save(&mut self, vs: &VarStore, completed: CompletedUnit<P>) -> Result<()> {
        match completed {
            CompletedUnit::Epoch(_) => {
                vs.save(format!("{}epoch-{}", self.prefix, self.epoch))
                    .context(format!(
                        "Cannot save variable store for epoch {}",
                        self.epoch
                    ))?;

                self.epoch += 1;
                self.epoch_batch = 0;
            }
            CompletedUnit::Batch(_) => {
                if (self.batch + 1) % self.n_batches == 0 {
                    vs.save(format!(
                        "{}epoch-{}-batch-{}",
                        self.prefix, self.epoch, self.epoch_batch
                    ))
                    .context(format!(
                        "Cannot save variable store for epoch {} batch {}",
                        self.epoch, self.batch
                    ))?;
                }

                self.batch += 1;
                self.epoch_batch += 1;
            }
        }

        Ok(())
    }
}
