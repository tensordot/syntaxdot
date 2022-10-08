use std::io::BufWriter;

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command};
use conllu::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, Output};
use syntaxdot::tagger::Tagger;
use syntaxdot_tokenizers::Tokenize;
use tch::{self, Device};

use crate::io::Model;
use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::SyntaxDotApp;

const CONFIG: &str = "CONFIG";
const GPU: &str = "GPU";
const INPUT: &str = "INPUT";
const MAX_BATCH_PIECES: &str = "MAX_BATCH_PIECES";
const MAX_LEN: &str = "MAX_LEN";
const NUM_ANNOTATION_THREADS: &str = "NUM_ANNOTATION_THREADS";
const NUM_INTEROP_THREADS: &str = "NUM_INTEROP_THREADS";
const NUM_INTRAOP_THREADS: &str = "NUM_INTRAOP_THREADS";
const OUTPUT: &str = "OUTPUT";
const READ_AHEAD: &str = "READ_AHEAD";

pub struct AnnotateApp {
    config: String,
    device: Device,
    input: Option<String>,
    max_batch_pieces: usize,
    max_len: Option<usize>,
    num_annotation_threads: usize,
    num_interop_threads: usize,
    num_intraop_threads: usize,
    output: Option<String>,
    read_ahead: usize,
}

impl AnnotateApp {
    fn process<R, W>(
        &self,
        tokenizer: &dyn Tokenize,
        tagger: Tagger,
        read: R,
        write: W,
    ) -> Result<()>
    where
        R: ReadSentence,
        W: WriteSentence,
    {
        let mut speed = TaggerSpeed::new();

        let mut sent_proc = SentProcessor::new(
            &tagger,
            write,
            self.max_batch_pieces,
            self.max_len,
            self.read_ahead,
        );

        for sentence in read.sentences() {
            let sentence = sentence.context("Cannot parse sentence")?;

            let tokenized_sentence = tokenizer.tokenize(sentence);

            speed.count_sentence(&tokenized_sentence);

            sent_proc
                .process(tokenized_sentence)
                .context("Error processing sentence")?;
        }

        Ok(())
    }
}

impl SyntaxDotApp for AnnotateApp {
    fn app() -> Command {
        Command::new("annotate")
            .arg_required_else_help(true)
            .about("Annotate a corpus")
            .arg(
                Arg::new(CONFIG)
                    .help("SyntaxDot configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::new(INPUT).help("Input data").index(2))
            .arg(Arg::new(OUTPUT).help("Output data").index(3).num_args(1))
            .arg(
                Arg::new(GPU)
                    .long("gpu")
                    .num_args(1)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::new(MAX_BATCH_PIECES)
                    .long("max-batch-pieces")
                    .num_args(1)
                    .help("Maximum number of pieces per batch")
                    .default_value("1000"),
            )
            .arg(
                Arg::new(NUM_ANNOTATION_THREADS)
                    .help("Annotation threads")
                    .long("annotation-threads")
                    .value_name("N")
                    .default_value("4"),
            )
            .arg(
                Arg::new(NUM_INTEROP_THREADS)
                    .help("Inter op parallelism threads")
                    .long("interop-threads")
                    .value_name("N")
                    .default_value("1"),
            )
            .arg(
                Arg::new(NUM_INTRAOP_THREADS)
                    .help("Intra op parallelism threads")
                    .long("intraop-threads")
                    .value_name("N")
                    .default_value("1"),
            )
            .arg(
                Arg::new(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .num_args(1)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::new(READ_AHEAD)
                    .help("Readahead (number of sentences)")
                    .long("readahead")
                    .default_value("5000"),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.get_one::<String>(CONFIG).unwrap().into();
        let device = match matches.get_one::<String>("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .context(format!("Cannot parse GPU number ({})", gpu))?,
            ),
            None => Device::Cpu,
        };
        let input = matches.get_one::<String>(INPUT).map(ToOwned::to_owned);
        let max_batch_pieces = matches
            .get_one::<String>(MAX_BATCH_PIECES)
            .unwrap()
            .parse()
            .context("Cannot parse maximum number of batch pieces")?;
        let num_annotation_threads = matches
            .get_one::<String>(NUM_ANNOTATION_THREADS)
            .unwrap()
            .parse()
            .context("Cannot number of inter op threads")?;
        let num_interop_threads = matches
            .get_one::<String>(NUM_INTEROP_THREADS)
            .unwrap()
            .parse()
            .context("Cannot number of inter op threads")?;
        let num_intraop_threads = matches
            .get_one::<String>(NUM_INTRAOP_THREADS)
            .unwrap()
            .parse()
            .context("Cannot number of intra op threads")?;
        let max_len = matches
            .get_one::<String>(MAX_LEN)
            .map(|v| v.parse().context("Cannot parse maximum sentence length"))
            .transpose()?;
        let output = matches.get_one::<String>(OUTPUT).map(ToOwned::to_owned);
        let read_ahead = matches
            .get_one::<String>(READ_AHEAD)
            .unwrap()
            .parse()
            .context("Cannot parse number of sentences to read ahead")?;

        Ok(AnnotateApp {
            config,
            device,
            input,
            max_batch_pieces,
            max_len,
            num_annotation_threads,
            num_interop_threads,
            num_intraop_threads,
            output,
            read_ahead,
        })
    }

    fn run(&self) -> Result<()> {
        // Set number of PyTorch threads.
        tch::set_num_threads(self.num_intraop_threads as i32);
        tch::set_num_interop_threads(self.num_interop_threads as i32);

        // Rayon threads.
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_annotation_threads)
            .build_global()
            .unwrap();

        let model = Model::load(&self.config, self.device, true, false, |_| 0)?;
        let tagger = Tagger::new(
            self.device,
            model.model,
            model.biaffine_encoder,
            model.encoders,
        );

        let input = Input::from(self.input.as_ref());
        let reader = Reader::new(input.buf_read().context("Cannot open input for reading")?);

        let output = Output::from(self.output.as_ref());
        let writer = Writer::new(BufWriter::new(
            output.write().context("Cannot open output for writing")?,
        ));

        self.process(&*model.tokenizer, tagger, reader, writer)
    }
}
