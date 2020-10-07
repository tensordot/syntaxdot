use anyhow::Result;

pub trait SummaryWriter {
    fn write_scalar(&self, tag: &str, step: i64, value: f32) -> Result<()>;
}

pub struct NoopWriter;

impl SummaryWriter for NoopWriter {
    fn write_scalar(&self, _tag: &str, _step: i64, _value: f32) -> Result<()> {
        Ok(())
    }
}

pub struct SummaryOption;

#[cfg(feature = "tensorboard")]
pub(crate) mod tensorboard;
#[cfg(feature = "tensorboard")]
mod option_impl {
    use anyhow::Result;
    use clap::{App, Arg, ArgMatches};

    use super::{tensorboard, NoopWriter, SummaryOption, SummaryWriter};
    use crate::traits::SyntaxDotOption;

    const LOG_PREFIX: &str = "LOG_PREFIX";

    impl SyntaxDotOption for SummaryOption {
        type Value = Box<dyn SummaryWriter>;

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
                    Box::new(tensorboard::TensorBoardWriter::new(prefix)?) as Box<dyn SummaryWriter>
                }
                None => Box::new(NoopWriter),
            })
        }
    }
}

#[cfg(not(feature = "tensorboard"))]
mod option_impl {
    use anyhow::Result;
    use clap::{App, ArgMatches};

    use super::{NoopWriter, SummaryOption, SummaryWriter};
    use crate::traits::SyntaxDotOption;

    impl SyntaxDotOption for SummaryOption {
        type Value = Box<dyn SummaryWriter>;

        fn add_to_app(app: App<'static, 'static>) -> App<'static, 'static> {
            app
        }

        fn parse(_matches: &ArgMatches) -> Result<Self::Value> {
            Ok(Box::new(NoopWriter))
        }
    }
}
