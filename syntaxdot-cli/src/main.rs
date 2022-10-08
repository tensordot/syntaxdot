use std::io::stdout;

use anyhow::Result;
use clap::{builder::EnumValueParser, crate_version, App, AppSettings, Arg, SubCommand};
use clap_complete::{generate, Shell};

pub mod io;

pub mod progress;

pub mod save;

pub mod sent_proc;

pub mod summary;

mod subcommands;

pub mod traits;
use traits::SyntaxDotApp;

pub mod util;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
    AppSettings::SubcommandRequiredElseHelp,
];

fn main() -> Result<()> {
    // Known subapplications.
    let apps = vec![
        subcommands::AnnotateApp::app(),
        subcommands::DistillApp::app(),
        subcommands::FilterLenApp::app(),
        subcommands::FinetuneApp::app(),
        subcommands::PrepareApp::app(),
    ];

    env_logger::init();

    let cli = App::new("syntaxdot")
        .settings(DEFAULT_CLAP_SETTINGS)
        .about("A neural sequence labeler")
        .version(crate_version!())
        .subcommands(apps)
        .subcommand(
            SubCommand::with_name("completions")
                .about("Generate completion scripts for your shell")
                .setting(AppSettings::ArgRequiredElseHelp)
                .arg(Arg::with_name("shell").value_parser(EnumValueParser::<Shell>::new())),
        );
    let matches = cli.clone().get_matches();

    match matches.subcommand_name().unwrap() {
        "annotate" => {
            subcommands::AnnotateApp::parse(matches.subcommand_matches("annotate").unwrap())?.run()
        }
        "completions" => {
            let shell = matches
                .subcommand_matches("completions")
                .unwrap()
                .get_one::<Shell>("shell")
                .unwrap();
            write_completion_script(cli, *shell);
            Ok(())
        }
        "distill" => {
            subcommands::DistillApp::parse(matches.subcommand_matches("distill").unwrap())?.run()
        }
        "finetune" => {
            subcommands::FinetuneApp::parse(matches.subcommand_matches("finetune").unwrap())?.run()
        }
        "filter-len" => {
            subcommands::FilterLenApp::parse(matches.subcommand_matches("filter-len").unwrap())?
                .run()
        }
        "prepare" => {
            subcommands::PrepareApp::parse(matches.subcommand_matches("prepare").unwrap())?.run()
        }
        _unknown => unreachable!(),
    }
}

fn write_completion_script(mut cli: App, shell: Shell) {
    generate(shell, &mut cli, "syntaxdot", &mut stdout());
}
