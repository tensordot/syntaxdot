// copyright 2018 the google ai language team authors and the huggingface inc. team.
// copyright (c) 2018, nvidia corporation.  all rights reserved.
// copyright (c) 2019 the sticker developers.
//
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// limitations under the license.

use std::collections::HashMap;

use tch::nn::VarStore;
use tch::Tensor;

use super::{Optimizer, ZeroGrad};

/// Internal Adam state.
struct AdamWState {
    step: usize,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

/// Adam optimizer configuration.
#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    pub betas: (f64, f64),
    pub correct_bias: bool,
    pub eps: f64,
    pub lr: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig {
            betas: (0.9, 0.999),
            correct_bias: false,
            eps: 1e-6,
            lr: 1e-3,
            weight_decay: 0.,
        }
    }
}

/// Adam algorithm with weight decay fix.
pub struct AdamW<'a> {
    state: HashMap<String, AdamWState>,
    vs: &'a VarStore,
}

impl<'a> AdamW<'a> {
    pub fn new(vs: &'a VarStore) -> Self {
        AdamW {
            state: HashMap::new(),
            vs,
        }
    }
}

impl<'a> Optimizer for AdamW<'a> {
    type Config = AdamWConfig;

    fn backward_step<F>(&mut self, loss: &Tensor, config_fun: F)
    where
        F: Fn(&str) -> Self::Config,
    {
        self.vs.zero_grad();
        loss.backward();
        tch::no_grad(|| self.step(config_fun));
    }

    fn step<F>(&mut self, config_fun: F)
    where
        F: Fn(&str) -> Self::Config,
    {
        for (name, mut tensor) in self.vs.variables() {
            if !tensor.grad().defined() {
                continue;
            }

            let config = config_fun(&name);

            let grad = tensor.grad();

            let mut state = self.state.entry(name.to_string()).or_insert(AdamWState {
                step: 0,
                exp_avg: Tensor::zeros_like(&tensor),
                exp_avg_sq: Tensor::zeros_like(&tensor),
            });

            state.step += 1;

            // Decay the first and second moment running average coefficient
            // In-place operations to update the averages at the same time
            state.exp_avg *= config.betas.0;
            state.exp_avg += (1. - config.betas.0) * &grad;
            state.exp_avg_sq *= config.betas.1;
            state.exp_avg_sq += (1. - config.betas.1) * &grad * &grad;
            let mut denom = state.exp_avg_sq.sqrt();
            denom += config.eps;

            let mut step_size = config.lr;
            if config.correct_bias {
                let bias_correction1 = 1.0 - config.betas.0.powf(state.step as f64);
                let bias_correction2 = 1.0 - config.betas.1.powf(state.step as f64);
                step_size *= bias_correction2.sqrt() / bias_correction1;
            }

            tensor += -step_size * (&state.exp_avg / denom);

            if config.weight_decay > 0. {
                tensor += -config.lr * config.weight_decay * &tensor;
            }
        }
    }

    fn var_store(&self) -> &VarStore {
        self.vs
    }
}
