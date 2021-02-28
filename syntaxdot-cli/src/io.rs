use std::fs::File;

use anyhow::{Context, Result};
use syntaxdot::config::{BiaffineParserConfig, Config, PretrainConfig, TomlRead};
use syntaxdot::encoders::Encoders;
use syntaxdot::model::bert::BertModel;
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_tch_ext::RootExt;
use syntaxdot_tokenizers::Tokenize;
use tch::nn::VarStore;
use tch::Device;

/// Wrapper around different parts of a model.
pub struct Model {
    pub biaffine_encoder: Option<ImmutableDependencyEncoder>,
    pub encoders: Encoders,
    pub model: BertModel,
    pub pretrain_config: PretrainConfig,
    pub tokenizer: Box<dyn Tokenize>,
    pub vs: VarStore,
}

impl Model {
    /// Load a model on the given device.
    ///
    /// If `freeze` is true, gradient computation is disabled for the
    /// model parameters.
    ///
    /// If `load_partial` is true, a model file will be loaded successfully,
    /// even when not all model parameters are present. This is used for
    /// finetuning of pretrained models, which do not contain the classifier
    /// portion of the model.
    pub fn load<F>(
        config_path: &str,
        device: Device,
        freeze: bool,
        load_partial: bool,
        parameter_group_fun: F,
    ) -> Result<Self>
    where
        F: 'static + Fn(&str) -> usize,
    {
        let config = load_config(config_path)?;
        Self::load_from(
            config_path,
            &config.model.parameters,
            device,
            freeze,
            load_partial,
            parameter_group_fun,
        )
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// `parameters_path`.
    ///
    /// If `freeze` is true, gradient computation is disabled for the
    /// model parameters.
    ///
    /// If `load_partial` is true, a model file will be loaded successfully,
    /// even when not all model parameters are present. This is used for
    /// finetuning of pretrained models, which do not contain the classifier
    /// portion of the model.
    pub fn load_from<F>(
        config_path: &str,
        parameters_path: &str,
        device: Device,
        freeze: bool,
        load_partial: bool,
        parameter_group_fun: F,
    ) -> Result<Model>
    where
        F: 'static + Fn(&str) -> usize,
    {
        let config = load_config(config_path)?;
        let biaffine_decoder = config
            .biaffine
            .as_ref()
            .map(|config| load_biaffine_decoder(config))
            .transpose()?;
        let encoders = load_encoders(&config)?;
        let tokenizer = load_tokenizer(&config)?;
        let pretrain_config = load_pretrain_config(&config)?;

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root_ext(parameter_group_fun),
            &pretrain_config,
            config.biaffine.as_ref(),
            biaffine_decoder
                .as_ref()
                .map(ImmutableDependencyEncoder::n_relations)
                .unwrap_or(0),
            &encoders,
            config.model.pooler,
            0.0,
            config.model.position_embeddings,
        )
        .context("Cannot construct model")?;

        if load_partial {
            vs.load_partial(parameters_path)
                .context("Cannot load model parameters")?;
        } else {
            vs.load(parameters_path)
                .context("Cannot load model parameters")?;
        }

        if freeze {
            vs.freeze();
        }

        Ok(Model {
            biaffine_encoder: biaffine_decoder,
            encoders,
            model,
            pretrain_config,
            tokenizer,
            vs,
        })
    }
}

pub fn load_pretrain_config(config: &Config) -> Result<PretrainConfig> {
    config
        .model
        .pretrain_config()
        .context("Cannot load pretraining model configuration")
}

pub fn load_config(config_path: &str) -> Result<Config> {
    let config_file = File::open(config_path)
        .context(format!("Cannot open configuration file '{}'", &config_path))?;
    let mut config = Config::from_toml_read(config_file)
        .context(format!("Cannot parse configuration file: {}", config_path))?;
    config.relativize_paths(config_path).context(format!(
        "Cannot relativize paths in configuration file: {}",
        config_path
    ))?;

    Ok(config)
}

fn load_biaffine_decoder(config: &BiaffineParserConfig) -> Result<ImmutableDependencyEncoder> {
    let f = File::open(&config.labels).context(format!(
        "Cannot open dependency label file: {}",
        config.labels
    ))?;

    let encoder: ImmutableDependencyEncoder = serde_yaml::from_reader(&f).context(format!(
        "Cannot deserialize dependency labels from: {}",
        config.labels
    ))?;

    log::info!("Loaded biaffine encoder: {} labels", encoder.n_relations());

    Ok(encoder)
}

fn load_encoders(config: &Config) -> Result<Encoders> {
    let f = File::open(&config.labeler.labels)
        .context(format!("Cannot open label file: {}", config.labeler.labels))?;
    let encoders: Encoders = serde_yaml::from_reader(&f).context(format!(
        "Cannot deserialize labels from: {}",
        config.labeler.labels
    ))?;

    for encoder in &*encoders {
        log::info!(
            "Loaded labels for encoder '{}': {} labels",
            encoder.name(),
            encoder.encoder().len()
        );
    }

    Ok(encoders)
}

pub fn load_tokenizer(config: &Config) -> Result<Box<dyn Tokenize>> {
    config
        .tokenizer()
        .context("Cannot read tokenizer vocabulary")
}
