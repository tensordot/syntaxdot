use anyhow::Result;
use clap::{App, Arg, ArgMatches};

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

    fn add_to_app(app: App<'static, 'static>) -> App<'static, 'static> {
        app.arg(
            Arg::with_name(LOG_PREFIX)
                .long("log-prefix")
                .value_name("PREFIX")
                .takes_value(true)
                .help("Prefix for Tensorboard logs"),
        )
    }

    fn parse(matches: &ArgMatches) -> Result<Self::Value> {
        Ok(match matches.value_of(LOG_PREFIX) {
            Some(prefix) => {
                Box::new(tensorboard::TensorBoardWriter::new(prefix)?) as Box<dyn ScalarWriter>
            }
            None => Box::new(NoopWriter),
        })
    }
}
