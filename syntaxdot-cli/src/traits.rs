use anyhow::Result;
use clap::{App, AppSettings, ArgMatches};
use syntaxdot::optimizers::{GradScaler, Optimizer};
use tch::nn::{adamw, AdamW, Optimizer as TchOptimizer, OptimizerConfig, VarStore};

pub static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub enum ParameterGroup {
    Encoder = 0,
    Classifier = 1,
    EncoderNoWeightDecay = 2,
    ClassifierNoWeightDecay = 3,
}

pub trait SyntaxDotApp
where
    Self: Sized,
{
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self>;

    fn run(&self) -> Result<()>;
}

pub trait SyntaxDotTrainApp: SyntaxDotApp {
    fn build_parameter_group_fun() -> fn(&str) -> usize {
        |name: &str| {
            if name.starts_with("classifiers") || name.starts_with("biaffine") {
                if name.contains("layer_norm") || name.contains("bias") {
                    ParameterGroup::ClassifierNoWeightDecay as usize
                } else {
                    ParameterGroup::Classifier as usize
                }
            } else if name.starts_with("encoder") || name.starts_with("embeddings") {
                if name.contains("layer_norm") || name.contains("bias") {
                    ParameterGroup::EncoderNoWeightDecay as usize
                } else {
                    ParameterGroup::Encoder as usize
                }
            } else {
                unreachable!();
            }
        }
    }

    fn build_optimizer(&self, var_store: &VarStore) -> Result<GradScaler<TchOptimizer<AdamW>>> {
        let opt = adamw(0.9, 0.999, self.weight_decay()).build(var_store, 1e-3)?;
        let mut grad_scaler = GradScaler::new_with_defaults(self.mixed_precision(), opt)?;
        grad_scaler.set_weight_decay_group(ParameterGroup::EncoderNoWeightDecay as usize, 0.);
        grad_scaler.set_weight_decay_group(ParameterGroup::ClassifierNoWeightDecay as usize, 0.);
        Ok(grad_scaler)
    }

    fn mixed_precision(&self) -> bool;

    fn weight_decay(&self) -> f64;
}

pub trait SyntaxDotOption {
    type Value;

    fn add_to_app(app: App<'static, 'static>) -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self::Value>;
}
