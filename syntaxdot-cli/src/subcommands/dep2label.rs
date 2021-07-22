use std::io::BufWriter;

use anyhow::{bail, Context, Result};
use clap::{App, AppSettings, Arg, ArgMatches};
use conllu::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, Output};
use syntaxdot_encoders::depseq::{PosLayer, RelativePosEncoder, RelativePositionEncoder};
use syntaxdot_encoders::SentenceEncoder;
use udgraph::graph::Node;

use crate::SyntaxDotApp;

static ENCODER: &str = "ENCODER";
static INPUT: &str = "INPUT";
static LABEL_FEATURE: &str = "LABEL_FEATURE";
static POS_LAYER: &str = "POS_LAYER";
static OUTPUT: &str = "OUTPUT";
static ROOT_LABEL: &str = "ROOT_LABEL";

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub struct Dep2LabelApp {
    encoder: String,
    input: Option<String>,
    label_feature: String,
    output: Option<String>,
    pos_layer: PosLayer,
    root_label: String,
}

impl SyntaxDotApp for Dep2LabelApp {
    fn app() -> App<'static, 'static> {
        App::new("dep2label")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Convert dependencies to labels")
            .arg(
                Arg::with_name(ENCODER)
                    .short("e")
                    .long("encoder")
                    .value_name("ENC")
                    .help("Dependency encoder")
                    .possible_values(&["pos", "position"])
                    .default_value("pos"),
            )
            .arg(
                Arg::with_name(LABEL_FEATURE)
                    .short("f")
                    .long("feature")
                    .value_name("NAME")
                    .help("Name of the feature used for the dependency label")
                    .default_value("deplabel"),
            )
            .arg(
                Arg::with_name(POS_LAYER)
                    .short("p")
                    .long("pos-layer")
                    .value_name("LAYER")
                    .help("Part-of-speech tag layer for relative positions")
                    .possible_values(&["upos", "xpos"])
                    .default_value("upos"),
            )
            .arg(
                Arg::with_name(ROOT_LABEL)
                    .short("r")
                    .long("root")
                    .value_name("LABEL")
                    .help("Root dependency label")
                    .default_value("root"),
            )
            .arg(Arg::with_name(INPUT).help("Input data").index(1))
            .arg(Arg::with_name(OUTPUT).help("Output data").index(2))
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let encoder = matches.value_of(ENCODER).unwrap().into();
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let label_feature = matches.value_of(LABEL_FEATURE).unwrap().into();
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);
        let root_label = matches.value_of(ROOT_LABEL).unwrap().into();

        let pos_layer = match matches.value_of(POS_LAYER).unwrap() {
            "upos" => PosLayer::UPos,
            "xpos" => PosLayer::XPos,
            unknown => bail!("Unknown Part-of-speech tags layer: {}", unknown),
        };

        Ok(Dep2LabelApp {
            encoder,
            input,
            label_feature,
            output,
            pos_layer,
            root_label,
        })
    }

    fn run(&self) -> Result<()> {
        let input = Input::from(self.input.as_ref());
        let reader = Reader::new(input.buf_read().context("Cannot open input for reading")?);

        let output = Output::from(self.output.as_ref());
        let writer = Writer::new(BufWriter::new(
            output.write().context("Cannot open output for writing")?,
        ));

        match self.encoder.as_str() {
            "pos" => self.label_with_encoder(
                RelativePosEncoder::new(self.pos_layer, &self.root_label),
                reader,
                writer,
            ),
            "position" => self.label_with_encoder(
                RelativePositionEncoder::new(&self.root_label),
                reader,
                writer,
            ),
            unknown => {
                bail!("Unknown encoder: {}", unknown);
            }
        }
    }
}

impl Dep2LabelApp {
    fn label_with_encoder<E, R, W>(&self, encoder: E, read: R, mut write: W) -> Result<()>
    where
        E: SentenceEncoder,
        E::Encoding: ToString,
        E::Error: 'static + Send + Sync,
        R: ReadSentence,
        W: WriteSentence,
    {
        for sentence in read.sentences() {
            let mut sentence = sentence.context("Cannot parse sentence")?;

            let encoded = encoder
                .encode(&sentence)
                .context("Cannot dependency-encode sentence")?;

            for (token, encoding) in sentence.iter_mut().filter_map(Node::token_mut).zip(encoded) {
                token
                    .misc_mut()
                    .insert(self.label_feature.clone(), Some(encoding.to_string()));
            }

            write
                .write_sentence(&sentence)
                .context("Cannot write sentence")?;
        }

        Ok(())
    }
}
