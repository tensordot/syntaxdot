use std::fs::File;

use anyhow::{Context, Result};
use syntaxdot::config::{Config, PretrainConfig, TomlRead};
use syntaxdot::encoders::Encoders;
use syntaxdot::model::bert::BertModel;
use syntaxdot_tch_ext::RootExt;
use syntaxdot_tokenizers::Tokenize;
use tch::nn::VarStore;
use tch::Device;

/// Wrapper around different parts of a model.
pub struct Model {
    pub encoders: Encoders,
    pub model: BertModel,
    pub tokenizer: Box<dyn Tokenize>,
    pub vs: VarStore,
}

impl Model {
    /// Load a model on the given device.
    ///
    /// If `freeze` is true, gradient computation is disabled for the
    /// model parameters.
    pub fn load<F>(
        config_path: &str,
        device: Device,
        freeze: bool,
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
    pub fn load_from<F>(
        config_path: &str,
        parameters_path: &str,
        device: Device,
        freeze: bool,
        parameter_group_fun: F,
    ) -> Result<Model>
    where
        F: 'static + Fn(&str) -> usize,
    {
        let config = load_config(config_path)?;
        let encoders = load_encoders(&config)?;
        let tokenizer = load_tokenizer(&config)?;
        let pretrain_config = load_pretrain_config(&config)?;

        let mut vs = VarStore::new(device);

        let model = BertModel::new(
            vs.root_ext(parameter_group_fun),
            &pretrain_config,
            &encoders,
            0.0,
            config.model.position_embeddings,
        )
        .context("Cannot construct model")?;

        vs.load(parameters_path)
            .context("Cannot load model parameters")?;

        if freeze {
            vs.freeze();
        }

        Ok(Model {
            encoders,
            model,
            tokenizer,
            vs,
        })
    }

    /// Load a model on the given device.
    ///
    /// In contrast to `load_model`, this does not load the parameters
    /// specified in the configuration file, but the parameters from
    /// the HDF5 file at `hdf5_path`.
    #[cfg(feature = "load-hdf5")]
    pub fn load_from_hdf5<F>(
        config_path: &str,
        hdf5_path: &str,
        device: Device,
        parameter_group_fun: F,
    ) -> Result<Model>
    where
        F: 'static + Fn(&str) -> usize,
    {
        let config = load_config(config_path)?;
        let encoders = load_encoders(&config)?;
        let tokenizer = load_tokenizer(&config)?;
        let pretrain_config = load_pretrain_config(&config)?;

        let vs = VarStore::new(device);

        let model = BertModel::from_pretrained(
            vs.root_ext(parameter_group_fun),
            &pretrain_config,
            hdf5_path,
            &encoders,
            0.5,
        )
        .context("Cannot load pretrained model parameters")?;

        Ok(Model {
            encoders,
            model,
            tokenizer,
            vs,
        })
    }

    #[cfg(not(feature = "load-hdf5"))]
    pub fn load_from_hdf5<F>(
        _config_path: &str,
        _hdf5_path: &str,
        _device: Device,
        _parameter_group_fun: F,
    ) -> Result<Model> {
        anyhow::bail!("Cannot load HDF5 model: SyntaxDot was compiled without support for HDF5");
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

fn load_encoders(config: &Config) -> Result<Encoders> {
    let f = File::open(&config.labeler.labels)
        .context(format!("Cannot open label file: {}", config.labeler.labels))?;
    let encoders: Encoders = serde_yaml::from_reader(&f).context(format!(
        "Cannot deserialize labels from: {}",
        config.labeler.labels
    ))?;

    for encoder in &*encoders {
        eprintln!(
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
