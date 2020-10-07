use tch::nn::VarStore;
use tch::{Kind, Tensor};

use super::{Optimizer, ZeroGrad};

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
    ) -> Self {
        let device = optimizer.var_store().device();

        GradScaler {
            enabled,
            growth_factor,
            backoff_factor,
            growth_interval,

            optimizer,

            found_inf: Tensor::full(&[1], 0.0, (Kind::Float, device)),
            growth_tracker: Tensor::full(&[1], 0, (Kind::Int, device)),
            scale: Tensor::full(&[1], init_scale, (Kind::Float, device)),
        }
    }

    pub fn new_with_defaults(enabled: bool, optimizer: O) -> Self {
        GradScaler::new(enabled, optimizer, 2f64.powi(16), 2., 0.5, 2000)
    }

    pub fn current_scale(&self) -> f32 {
        Vec::<f32>::from(&self.scale)[0]
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    pub fn scale(&mut self, t: &Tensor) -> Tensor {
        if !self.enabled {
            t.shallow_clone()
        } else {
            t * &self.scale
        }
    }

    pub fn update(&mut self) {
        if !self.enabled {
            return;
        };

        self.scale = Tensor::internal_amp_update_scale(
            &self.growth_tracker,
            &self.scale,
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
    type Config = O::Config;

    fn backward_step<F>(&mut self, loss: &Tensor, config_fun: F)
    where
        F: Fn(&str) -> Self::Config,
    {
        self.var_store().zero_grad();
        self.scale(loss).backward();
        tch::no_grad(|| self.step(config_fun));
        self.update();
    }

    fn step<F>(&mut self, config_fun: F)
    where
        F: Fn(&str) -> Self::Config,
    {
        if !self.enabled {
            return self.optimizer.step(config_fun);
        }

        let inv_scale = self.scale.reciprocal().to_kind(Kind::Float);

        for (_, tensor) in self.optimizer.var_store().variables() {
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
            self.optimizer.step(config_fun)
        }
    }

    fn var_store(&self) -> &VarStore {
        self.optimizer.var_store()
    }
}
