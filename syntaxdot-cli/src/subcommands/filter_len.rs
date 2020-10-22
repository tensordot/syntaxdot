use std::io::BufWriter;

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use conllu::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, Output};

use crate::io::{load_config, load_tokenizer};
use crate::traits::{SyntaxDotApp, DEFAULT_CLAP_SETTINGS};

const CONFIG: &str = "CONFIG";
const MAX_LEN: &str = "MAX_LEN";
const INPUT: &str = "INPUT";
const OUTPUT: &str = "OUTPUT";

pub struct FilterLenApp {
    config: String,
    input: Option<String>,
    max_len: usize,
    output: Option<String>,
}

impl SyntaxDotApp for FilterLenApp {
    fn app() -> App<'static, 'static> {
        App::new("filter-len")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Filter corpus by the sentence length in pieces")
            .arg(
                Arg::with_name(CONFIG)
                    .help("SyntaxDot configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .help("Maximum sentence length")
                    .index(2)
                    .required(true),
            )
            .arg(Arg::with_name(INPUT).help("Input corpus").index(3))
            .arg(Arg::with_name(OUTPUT).help("Output corpus").index(4))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.value_of(CONFIG).unwrap().into();
        let max_len = matches
            .value_of(MAX_LEN)
            .unwrap()
            .parse()
            .context("Cannot parse maximum sentence length")?;
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);

        Ok(FilterLenApp {
            config,
            max_len,
            input,
            output,
        })
    }

    fn run(&self) -> Result<()> {
        let config = load_config(&self.config)?;

        let tokenizer = load_tokenizer(&config)?;

        let input = Input::from(self.input.as_ref());
        let output = Output::from(self.output.as_ref());

        let treebank_reader = Reader::new(
            input
                .buf_read()
                .context("Cannot open treebank for reading")?,
        );

        let mut treebank_writer = Writer::new(BufWriter::new(
            output.write().context("Cannot open treebank for writing")?,
        ));

        for sentence in treebank_reader.sentences() {
            let sentence = sentence.context("Cannot read sentence from treebank")?;

            let sentence_with_pieces = tokenizer.tokenize(sentence);

            if sentence_with_pieces.pieces().len() <= self.max_len {
                treebank_writer
                    .write_sentence(sentence_with_pieces.sentence())
                    .context("Cannot write sentence")?;
            }
        }

        Ok(())
    }
}
