use anyhow::Result;
use clap::{App, AppSettings, ArgMatches};

pub static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub trait SyntaxDotApp
where
    Self: Sized,
{
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self>;

    fn run(&self) -> Result<()>;
}

pub trait SyntaxDotOption {
    type Value;

    fn add_to_app(app: App<'static, 'static>) -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self::Value>;
}
