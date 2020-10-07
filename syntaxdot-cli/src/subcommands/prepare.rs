use std::fs::File;
use std::io::{BufReader, Write};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use conllu::io::{ReadSentence, Reader};
use indicatif::ProgressStyle;
use syntaxdot::config::Config;
use syntaxdot::encoders::Encoders;
use syntaxdot_encoders::SentenceEncoder;

use crate::io::load_config;
use crate::progress::ReadProgress;
use crate::traits::{SyntaxDotApp, DEFAULT_CLAP_SETTINGS};

const CONFIG: &str = "CONFIG";
static TRAIN_DATA: &str = "TRAIN_DATA";

pub struct PrepareApp {
    config: String,
    train_data: String,
}

impl PrepareApp {
    fn write_labels(config: &Config, encoders: &Encoders) -> Result<()> {
        let mut f = File::create(&config.labeler.labels).context(format!(
            "Cannot create label file: {}",
            config.labeler.labels
        ))?;
        let serialized_labels =
            serde_yaml::to_string(&encoders).context("Cannot serialize labels")?;
        f.write_all(serialized_labels.as_bytes())
            .context("Cannot write labels")
    }
}

impl SyntaxDotApp for PrepareApp {
    fn app() -> App<'static, 'static> {
        App::new("prepare")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Prepare shape and label files for training")
            .arg(
                Arg::with_name(CONFIG)
                    .help("SyntaxDot configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(TRAIN_DATA)
                    .help("Training data")
                    .index(2)
                    .required(true),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.value_of(CONFIG).unwrap().into();
        let train_data = matches.value_of(TRAIN_DATA).unwrap().into();

        Ok(PrepareApp { config, train_data })
    }

    fn run(&self) -> Result<()> {
        let config = load_config(&self.config)?;

        let encoders: Encoders = (&config.labeler.encoders).into();

        let train_file = File::open(&self.train_data)
            .context(format!("Cannot open train data file: {}", self.train_data))?;
        let read_progress = ReadProgress::new(train_file).context("Cannot create progress bar")?;
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% {msg}"),
        );

        let treebank_reader = Reader::new(BufReader::new(read_progress));

        for sentence in treebank_reader.sentences() {
            let sentence = sentence.context("Cannot read sentence from treebank")?;

            for encoder in &*encoders {
                encoder.encoder().encode(&sentence).context(format!(
                    "Cannot encode sentence with encoder {}",
                    encoder.name()
                ))?;
            }
        }

        Self::write_labels(&config, &encoders)
    }
}
