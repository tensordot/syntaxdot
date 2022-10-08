use anyhow::Result;
use clap::{Arg, ArgMatches, Command};

use crate::traits::SyntaxDotOption;

mod noop;
use noop::NoopWriter;

mod tensorboard;

const LOG_PREFIX: &str = "LOG_PREFIX";

pub trait ScalarWriter {
    fn write_scalar(&self, tag: &str, step: i64, value: f32) -> Result<()>;
}

pub struct SummaryOption;

impl SyntaxDotOption for SummaryOption {
    type Value = Box<dyn ScalarWriter>;

    fn add_to_app(app: Command<'static>) -> Command<'static> {
        app.arg(
            Arg::new(LOG_PREFIX)
                .long("log-prefix")
                .value_name("PREFIX")
                .takes_value(true)
                .help("Prefix for Tensorboard logs"),
        )
    }

    fn parse(matches: &ArgMatches) -> Result<Self::Value> {
        Ok(match matches.get_one::<String>(LOG_PREFIX) {
            Some(prefix) => {
                Box::new(tensorboard::TensorBoardWriter::new(prefix)?) as Box<dyn ScalarWriter>
            }
            None => Box::new(NoopWriter),
        })
    }
}
