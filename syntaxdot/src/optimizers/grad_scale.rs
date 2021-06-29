use tch::{Kind, Tensor};

use super::{Optimizer, ZeroGrad};
use crate::error::SyntaxDotError;

/// Gradient scaler
///
/// This data type implements gradient scaling.
///
/// In mixed-precision training, gradients underflow more quickly in FP16
/// as they become smaller, stopping backpropagation. Gradient scaling
/// counters this by scaling up the loss, to increase the magnitude of
/// gradients. The gradients are then unscaled in FP32. Since loss
/// scaling can also lead to overflow of gradients, the gradients are
/// checked for infinites before performing an optimizer step. If one or
/// more infinite gradients are found, the optimizer step is skipped and
/// the scale is reduced for the next step.
///
/// `GradientScaler` wraps an optimizer and implements the `Optimizer`
/// trait, so that it can be used in the same contexts as an optimizer
/// can be used.
pub struct GradScaler<O> {
    enabled: bool,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: i64,

    optimizer: O,

    found_inf: Tensor,
    growth_tracker: Tensor,
    scale: Tensor,
}

impl<O> GradScaler<O>
where
    O: Optimizer,
{
    fn new(
        enabled: bool,
        optimizer: O,
        init_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: i64,
    ) -> Result<Self, SyntaxDotError> {
        let device = match optimizer.trainable_variables().first() {
            Some(tensor) => tensor.device(),
            None => return Err(SyntaxDotError::NoTrainableVariables),
        };

        Ok(GradScaler {
            enabled,
            growth_factor,
            backoff_factor,
            growth_interval,

            optimizer,

            found_inf: Tensor::full(&[1], 0.0, (Kind::Float, device)),
            growth_tracker: Tensor::full(&[1], 0, (Kind::Int, device)),
            scale: Tensor::full(&[1], init_scale, (Kind::Float, device)),
        })
    }

    /// Construct a new gradient scaler.
    ///
    /// The gradient scaler wraps the given optimizer.
    pub fn new_with_defaults(enabled: bool, optimizer: O) -> Result<Self, SyntaxDotError> {
        GradScaler::new(enabled, optimizer, 2f64.powi(16), 2., 0.5, 2000)
    }

    /// Get the current scale.
    pub fn current_scale(&self) -> f32 {
        Vec::<f32>::from(&self.scale)[0]
    }

    /// Get a reference to the wrapped optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get a mutable reference to the wrapped optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Scale the given tensor.
    fn scale(&mut self, t: &Tensor) -> Result<Tensor, SyntaxDotError> {
        Ok(if !self.enabled {
            t.shallow_clone()
        } else {
            t.f_mul(&self.scale)?
        })
    }

    /// Update the scale for the next step.
    fn update(&mut self) {
        if !self.enabled {
            return;
        };

        self.scale = self.scale.internal_amp_update_scale_(
            &self.growth_tracker,
            &self.found_inf,
            self.growth_factor,
            self.backoff_factor,
            self.growth_interval,
        );

        // Clear infinity found status.
        self.found_inf = self.found_inf.zeros_like();
    }
}

impl<O> Optimizer for GradScaler<O>
where
    O: Optimizer,
{
    fn backward_step(&mut self, loss: &Tensor) -> Result<(), SyntaxDotError> {
        self.optimizer.trainable_variables().zero_grad();
        self.scale(loss)?.backward();
        tch::no_grad(|| self.step());
        self.update();
        Ok(())
    }

    fn set_lr_group(&mut self, group: usize, learning_rate: f64) {
        self.optimizer.set_lr_group(group, learning_rate)
    }

    fn set_weight_decay_group(&mut self, group: usize, weight_decay: f64) {
        self.optimizer.set_weight_decay_group(group, weight_decay)
    }

    fn step(&mut self) {
        if !self.enabled {
            return self.optimizer.step();
        }

        let inv_scale = self.scale.reciprocal().to_kind(Kind::Float);

        for tensor in &mut self.optimizer.trainable_variables() {
            if !tensor.grad().defined() {
                continue;
            }

            tensor
                .grad()
                .internal_amp_non_finite_check_and_unscale(&mut self.found_inf, &inv_scale);
        }

        let found_inf = (f32::from(&self.found_inf) - 1.0).abs() < f32::EPSILON;

        // Only step when there are no infinite gradients.
        if !found_inf {
            self.optimizer.step()
        }
    }

    fn trainable_variables(&self) -> Vec<Tensor> {
        self.optimizer.trainable_variables()
    }
}
