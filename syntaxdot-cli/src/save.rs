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
