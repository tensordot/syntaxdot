use std::io::BufWriter;

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use conllu::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, Output};
use syntaxdot::tagger::Tagger;
use syntaxdot_tokenizers::Tokenize;
use tch::{self, Device};

use crate::io::Model;
use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::{SyntaxDotApp, DEFAULT_CLAP_SETTINGS};

const BATCH_SIZE: &str = "BATCH_SIZE";
const CONFIG: &str = "CONFIG";
const GPU: &str = "GPU";
const INPUT: &str = "INPUT";
const MAX_LEN: &str = "MAX_LEN";
const NUM_INTEROP_THREADS: &str = "NUM_INTEROP_THREADS";
const NUM_INTRAOP_THREADS: &str = "NUM_INTRAOP_THREADS";
const OUTPUT: &str = "OUTPUT";
const READ_AHEAD: &str = "READ_AHEAD";

pub struct AnnotateApp {
    batch_size: usize,
    config: String,
    device: Device,
    input: Option<String>,
    num_interop_threads: usize,
    num_intraop_threads: usize,
    max_len: Option<usize>,
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
            self.batch_size,
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
    fn app() -> App<'static, 'static> {
        App::new("annotate")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Annotate a corpus")
            .arg(
                Arg::with_name(CONFIG)
                    .help("SyntaxDot configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(Arg::with_name(INPUT).help("Input data").index(2))
            .arg(
                Arg::with_name(OUTPUT)
                    .help("Output data")
                    .index(3)
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(BATCH_SIZE)
                    .long("batch-size")
                    .takes_value(true)
                    .help("Batch size")
                    .default_value("32"),
            )
            .arg(
                Arg::with_name(GPU)
                    .long("gpu")
                    .takes_value(true)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::with_name(NUM_INTEROP_THREADS)
                    .help("Inter op parallelism threads")
                    .long("interop-threads")
                    .value_name("N")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(NUM_INTRAOP_THREADS)
                    .help("Intra op parallelism threads")
                    .long("intraop-threads")
                    .value_name("N")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(READ_AHEAD)
                    .help("Readahead (number of batches)")
                    .long("readahead")
                    .default_value("100"),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.value_of(CONFIG).unwrap().into();
        let batch_size = matches
            .value_of(BATCH_SIZE)
            .unwrap()
            .parse()
            .context("Cannot parse batch size")?;
        let device = match matches.value_of("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .context(format!("Cannot parse GPU number ({})", gpu))?,
            ),
            None => Device::Cpu,
        };
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let num_interop_threads = matches
            .value_of(NUM_INTEROP_THREADS)
            .unwrap()
            .parse()
            .context("Cannot number of inter op threads")?;
        let num_intraop_threads = matches
            .value_of(NUM_INTRAOP_THREADS)
            .unwrap()
            .parse()
            .context("Cannot number of intra op threads")?;
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().context("Cannot parse maximum sentence length"))
            .transpose()?;
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);
        let read_ahead = matches
            .value_of(READ_AHEAD)
            .unwrap()
            .parse()
            .context("Cannot parse number of batches to read ahead")?;

        Ok(AnnotateApp {
            batch_size,
            config,
            device,
            input,
            num_interop_threads,
            num_intraop_threads,
            max_len,
            output,
            read_ahead,
        })
    }

    fn run(&self) -> Result<()> {
        // Set number of PyTorch threads.
        tch::set_num_threads(self.num_intraop_threads as i32);
        tch::set_num_interop_threads(self.num_interop_threads as i32);

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
