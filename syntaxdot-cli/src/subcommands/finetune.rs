use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::fs::File;
use std::io::BufReader;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use indicatif::ProgressStyle;
use ordered_float::NotNan;
use syntaxdot::dataset::{
    BatchedTensors, ConlluDataSet, DataSet, SentenceIterTools, SequenceLength,
};
use syntaxdot::encoders::Encoders;
use syntaxdot::lr::{ExponentialDecay, LearningRateSchedule, PlateauLearningRate};
use syntaxdot::model::bert::{BertModel, FreezeLayers};
use syntaxdot::optimizers::{GradScaler, Optimizer};
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_tokenizers::Tokenize;
use tch::nn::{self};
use tch::{self, Device, Kind};

use crate::io::Model;
use crate::progress::ReadProgress;
use crate::save::{BestEpochSaver, CompletedUnit, Save};
use crate::summary::{ScalarWriter, SummaryOption};
use crate::traits::{ParameterGroup, SyntaxDotApp, SyntaxDotOption, SyntaxDotTrainApp};
use crate::util::autocast_or_preserve;

const BATCH_SIZE: &str = "BATCH_SIZE";
const CONFIG: &str = "CONFIG";
const CONTINUE: &str = "CONTINUE";
const GPU: &str = "GPU";
const FINETUNE_EMBEDS: &str = "FINETUNE_EMBEDS";
const INITIAL_LR_CLASSIFIER: &str = "INITIAL_LR_CLASSIFIER";
const INITIAL_LR_ENCODER: &str = "INITIAL_LR_ENCODER";
const LABEL_SMOOTHING: &str = "LABEL_SMOOTHING";
const MIXED_PRECISION: &str = "MIXED_PRECISION";
const KEEP_BEST_EPOCHS: &str = "KEEP_BEST_EPOCHS";
const LR_DECAY_RATE: &str = "LR_DECAY_RATE";
const LR_PATIENCE: &str = "LR_PATIENCE";
const LR_SCALE: &str = "LR_SCALE";
const MAX_LEN: &str = "MAX_LEN";
const PATIENCE: &str = "PATIENCE";
const PRETRAINED_MODEL: &str = "PRETRAINED_MODEL";
const TRAIN_DATA: &str = "TRAIN_DATA";
const VALIDATION_DATA: &str = "VALIDATION_DATA";
const WARMUP: &str = "WARMUP";
const WEIGHT_DECAY: &str = "WEIGHT_DECAY";

pub struct LrSchedule {
    pub initial_lr_encoder: NotNan<f32>,
    pub initial_lr_classifier: NotNan<f32>,
    pub lr_decay_rate: NotNan<f32>,
    pub lr_scale: NotNan<f32>,
    pub lr_patience: usize,
    pub warmup_steps: usize,
}

pub struct FinetuneApp {
    batch_size: usize,
    config: String,
    continue_finetune: bool,
    device: Device,
    finetune_embeds: bool,
    max_len: SequenceLength,
    label_smoothing: Option<f64>,
    mixed_precision: bool,
    summary_writer: Box<dyn ScalarWriter>,
    lr_schedule: LrSchedule,
    patience: usize,
    pretrained_model: String,
    saver: BestEpochSaver<f32>,
    train_data: String,
    validation_data: String,
    weight_decay: f64,
}

pub struct LearningRateSchedules {
    pub classifier: PlateauLearningRate<ExponentialDecay>,
    pub encoder: PlateauLearningRate<ExponentialDecay>,
}

struct BiaffineEpochStats {
    las: f32,
    ls: f32,
    uas: f32,
    head_loss: f32,
    relation_loss: f32,
}

struct EpochStats {
    biaffine: Option<BiaffineEpochStats>,
    encoder_accuracy: BTreeMap<String, f32>,
    encoder_loss: BTreeMap<String, f32>,
    n_tokens: i64,
}

impl FinetuneApp {
    pub fn lr_schedules(&self) -> LearningRateSchedules {
        let exp_decay = ExponentialDecay::new(
            self.lr_schedule.initial_lr_classifier.into_inner(),
            self.lr_schedule.lr_decay_rate.into_inner(),
            1,
            false,
            self.lr_schedule.warmup_steps,
        );

        let classifier = PlateauLearningRate::new(
            exp_decay,
            self.lr_schedule.lr_scale.into_inner(),
            self.lr_schedule.lr_patience,
        );

        let mut encoder = classifier.clone();
        encoder.set_initial_lr(self.lr_schedule.initial_lr_encoder.into_inner());

        LearningRateSchedules {
            classifier,
            encoder,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epoch(
        &self,
        biaffine_encoder: Option<&ImmutableDependencyEncoder>,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        mut grad_scaler: Option<&mut GradScaler<impl Optimizer>>,
        lr_schedulers: &mut LearningRateSchedules,
        global_step: &mut usize,
        epoch: usize,
    ) -> Result<f32> {
        let epoch_stats = self.run_epoch_steps(
            biaffine_encoder,
            encoders,
            tokenizer,
            model,
            file,
            &mut grad_scaler,
            lr_schedulers,
            global_step,
            epoch,
        )?;

        self.log_epoch_stats(global_step, epoch_stats, grad_scaler.is_some())
    }

    fn log_epoch_stats(
        &self,
        global_step: &usize,
        epoch_stats: EpochStats,
        train: bool,
    ) -> Result<f32> {
        let epoch_type = if train { "train" } else { "validation" };

        let mut accs = Vec::new();

        if let Some(biaffine_stats) = epoch_stats.biaffine {
            accs.push(biaffine_stats.las / epoch_stats.n_tokens as f32);

            log::info!(
                "biaffine head loss: {:.4}, rel loss: {:.4}, las: {:.4}, uas: {:.4}, ls: {:.4}",
                biaffine_stats.head_loss / epoch_stats.n_tokens as f32,
                biaffine_stats.relation_loss / epoch_stats.n_tokens as f32,
                biaffine_stats.las / epoch_stats.n_tokens as f32,
                biaffine_stats.uas / epoch_stats.n_tokens as f32,
                biaffine_stats.ls / epoch_stats.n_tokens as f32
            );

            self.summary_writer.write_scalar(
                &format!("loss:{},biaffine:head", epoch_type),
                *global_step as i64,
                biaffine_stats.head_loss,
            )?;

            self.summary_writer.write_scalar(
                &format!("loss:{},biaffine:relation", epoch_type),
                *global_step as i64,
                biaffine_stats.relation_loss,
            )?;

            self.summary_writer.write_scalar(
                &format!("las:{},biaffine", epoch_type),
                *global_step as i64,
                biaffine_stats.las / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                &format!("ls:{},biaffine", epoch_type),
                *global_step as i64,
                biaffine_stats.ls / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                &format!("uas:{},biaffine", epoch_type),
                *global_step as i64,
                biaffine_stats.uas / epoch_stats.n_tokens as f32,
            )?;
        }

        for (encoder_name, loss) in epoch_stats.encoder_loss {
            let acc = epoch_stats.encoder_accuracy[&encoder_name] / epoch_stats.n_tokens as f32;
            let loss = loss / epoch_stats.n_tokens as f32;

            log::info!("{} loss: {} accuracy: {:.4}", encoder_name, loss, acc);

            self.summary_writer.write_scalar(
                &format!("loss:{},layer:{}", epoch_type, &encoder_name),
                *global_step as i64,
                loss,
            )?;

            self.summary_writer.write_scalar(
                &format!("acc:{},layer:{}", epoch_type, &encoder_name),
                *global_step as i64,
                acc,
            )?;

            accs.push(acc);
        }

        Ok(accs.iter().sum::<f32>() / accs.len() as f32)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epoch_steps(
        &self,
        biaffine_encoder: Option<&ImmutableDependencyEncoder>,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        mut grad_scaler: &mut Option<&mut GradScaler<impl Optimizer>>,
        lr_schedulers: &mut LearningRateSchedules,
        global_step: &mut usize,
        epoch: usize,
    ) -> Result<EpochStats> {
        let epoch_type = if grad_scaler.is_some() {
            "train"
        } else {
            "validation"
        };

        let read_progress = ReadProgress::new(file).context("Cannot create progress bar")?;
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(&format!(
            "[Time: {{elapsed_precise}}, ETA: {{eta_precise}}] {{bar}} {{percent}}% {} {{msg}}",
            epoch_type
        ))?);

        let mut dataset = ConlluDataSet::new(BufReader::new(read_progress));

        let mut n_tokens = 0;

        // Freeze the encoder during the first epoch.
        let freeze_encoder = epoch == 0;

        let mut biaffine_las = 0f32;
        let mut biaffine_ls = 0f32;
        let mut biaffine_uas = 0f32;
        let mut biaffine_head_loss = 0f32;
        let mut biaffine_relation_loss = 0f32;
        let mut encoder_accuracy = BTreeMap::new();
        let mut encoder_loss = BTreeMap::new();

        for batch in dataset
            .tokenize(tokenizer)?
            .filter_by_len(self.max_len)
            .batched_tensors(biaffine_encoder, Some(encoders), self.batch_size)
        {
            let batch = batch?;

            let (lr_classifier, lr_encoder) = if epoch == 0 {
                (self.lr_schedule.initial_lr_classifier.into_inner(), 0.)
            } else {
                (
                    lr_schedulers
                        .classifier
                        .compute_step_learning_rate(*global_step),
                    lr_schedulers
                        .encoder
                        .compute_step_learning_rate(*global_step),
                )
            };

            let attention_mask = batch.seq_lens.attention_mask()?;

            let n_batch_tokens =
                i64::try_from(batch.token_spans.token_mask()?.f_sum(Kind::Int64)?)?;

            let model_loss = autocast_or_preserve(self.mixed_precision, || {
                model.loss(
                    &batch.inputs.to_device(self.device),
                    &attention_mask.to_device(self.device),
                    &batch.token_spans.to_device(self.device),
                    batch
                        .biaffine_encodings
                        .map(|tensors| tensors.to_device(self.device)),
                    &batch
                        .labels
                        .expect("Batch without labels.")
                        .into_iter()
                        .map(|(encoder_name, labels)| (encoder_name, labels.to_device(self.device)))
                        .collect(),
                    self.label_smoothing,
                    grad_scaler.is_some(),
                    FreezeLayers {
                        embeddings: !self.finetune_embeds || freeze_encoder,
                        encoder: freeze_encoder,
                        classifiers: grad_scaler.is_none(),
                    },
                )
            })?;

            n_tokens += n_batch_tokens;

            let scalar_loss: f32 = model_loss
                .seq_classifiers
                .summed_loss
                .f_sum(Kind::Float)?
                .try_into()?;

            if let Some(scaler) = &mut grad_scaler {
                let optimizer = scaler.optimizer_mut();

                optimizer.set_lr_group(ParameterGroup::Encoder as usize, lr_encoder.into());
                optimizer.set_lr_group(
                    ParameterGroup::EncoderNoWeightDecay as usize,
                    lr_encoder.into(),
                );
                optimizer.set_lr_group(ParameterGroup::Classifier as usize, lr_classifier.into());
                optimizer.set_lr_group(
                    ParameterGroup::ClassifierNoWeightDecay as usize,
                    lr_classifier.into(),
                );

                let mut loss = model_loss.seq_classifiers.summed_loss.f_sum(Kind::Float)?;
                if let Some(biaffine_loss) = model_loss.biaffine.as_ref() {
                    let _ = loss.f_add_(
                        &biaffine_loss
                            .head_loss
                            .f_add(&biaffine_loss.relation_loss)?,
                    )?;
                }

                scaler.backward_step(&loss)?;

                if epoch != 0 {
                    self.summary_writer.write_scalar(
                        "gradient_scale",
                        *global_step as i64,
                        scaler.current_scale(),
                    )?;
                }

                if epoch != 0 {
                    *global_step += 1;
                }
            };

            for (encoder_name, loss) in model_loss.seq_classifiers.encoder_losses {
                *encoder_accuracy.entry(encoder_name.clone()).or_insert(0f32) +=
                    f32::try_from(&model_loss.seq_classifiers.encoder_accuracies[&encoder_name])?
                        * n_batch_tokens as f32;
                *encoder_loss.entry(encoder_name).or_insert(0f32) +=
                    f32::try_from(loss)? * n_batch_tokens as f32;
            }

            if let Some(biaffine_loss) = model_loss.biaffine.as_ref() {
                let head_loss = f32::try_from(&biaffine_loss.head_loss)?;
                let relation_loss = f32::try_from(&biaffine_loss.relation_loss)?;

                biaffine_las += f32::try_from(&biaffine_loss.acc.las)? * n_batch_tokens as f32;
                biaffine_ls += f32::try_from(&biaffine_loss.acc.ls)? * n_batch_tokens as f32;
                biaffine_uas += f32::try_from(&biaffine_loss.acc.uas)? * n_batch_tokens as f32;
                biaffine_head_loss += head_loss * n_batch_tokens as f32;
                biaffine_relation_loss += relation_loss * n_batch_tokens as f32;

                progress_bar.set_message(format!(
                    "classifier lr: {:.1e}, encoder lr: {:.1e}, seq loss: {:.4}, head loss: {:.4}, rel loss: {:.4}, global step: {}",
                    lr_classifier, lr_encoder, scalar_loss, head_loss, relation_loss, global_step
                ));
            } else {
                progress_bar.set_message(format!(
                    "classifier lr: {:.1e}, encoder lr: {:.1e}, seq loss: {:.4}, global step: {}",
                    lr_classifier, lr_encoder, scalar_loss, global_step
                ));
            }
        }

        progress_bar.finish();

        let biaffine_stats = biaffine_encoder.map(|_| BiaffineEpochStats {
            las: biaffine_las,
            ls: biaffine_ls,
            uas: biaffine_uas,
            head_loss: biaffine_head_loss,
            relation_loss: biaffine_relation_loss,
        });

        Ok(EpochStats {
            biaffine: biaffine_stats,
            encoder_accuracy,
            encoder_loss,
            n_tokens,
        })
    }
}

impl SyntaxDotApp for FinetuneApp {
    fn app() -> Command {
        let app = Command::new("finetune")
            .arg_required_else_help(true)
            .about("Finetune a model")
            .arg(
                Arg::new(CONFIG)
                    .help("SyntaxDot configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::new(CONTINUE)
                    .long("continue")
                    .action(ArgAction::SetTrue)
                    .help("Continue training a SyntaxDot model"),
            )
            .arg(
                Arg::new(PRETRAINED_MODEL)
                    .help("Pretrained model in Torch format")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::new(TRAIN_DATA)
                    .help("Training data")
                    .index(3)
                    .required(true),
            )
            .arg(
                Arg::new(VALIDATION_DATA)
                    .help("Validation data")
                    .index(4)
                    .required(true),
            )
            .arg(
                Arg::new(BATCH_SIZE)
                    .long("batch-size")
                    .num_args(1)
                    .help("Batch size")
                    .default_value("32"),
            )
            .arg(
                Arg::new(FINETUNE_EMBEDS)
                    .long("finetune-embeds")
                    .action(ArgAction::SetTrue)
                    .help("Finetune embeddings"),
            )
            .arg(
                Arg::new(GPU)
                    .long("gpu")
                    .num_args(1)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::new(INITIAL_LR_CLASSIFIER)
                    .long("lr-classifier")
                    .value_name("LR")
                    .help("Initial classifier learning rate")
                    .default_value("1e-3"),
            )
            .arg(
                Arg::new(INITIAL_LR_ENCODER)
                    .long("lr-encoder")
                    .value_name("LR")
                    .help("Initial encoder learning rate")
                    .default_value("5e-5"),
            )
            .arg(
                Arg::new(KEEP_BEST_EPOCHS)
                    .long("keep-best")
                    .value_name("N")
                    .help("Only keep the N best epochs"),
            )
            .arg(
                Arg::new(LABEL_SMOOTHING)
                    .long("label-smoothing")
                    .value_name("PROB")
                    .num_args(1)
                    .help("Distribute the given probability to non-target labels"),
            )
            .arg(
                Arg::new(MIXED_PRECISION)
                    .long("mixed-precision")
                    .action(ArgAction::SetTrue)
                    .help("Enable automatic mixed-precision"),
            )
            .arg(
                Arg::new(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .num_args(1)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::new(LR_DECAY_RATE)
                    .long("lr-decay-rate")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("0.99998"),
            )
            .arg(
                Arg::new(LR_PATIENCE)
                    .long("lr-patience")
                    .value_name("N")
                    .help("Scale learning rate after N epochs without improvement")
                    .default_value("2"),
            )
            .arg(
                Arg::new(LR_SCALE)
                    .long("lr-scale")
                    .value_name("SCALE")
                    .help("Value to scale the learning rate by")
                    .default_value("0.9"),
            )
            .arg(
                Arg::new(PATIENCE)
                    .long("patience")
                    .value_name("N")
                    .help("Maximum number of epochs without improvement")
                    .default_value("15"),
            )
            .arg(
                Arg::new(WARMUP)
                    .long("warmup")
                    .value_name("N")
                    .help(
                        "For the first N timesteps, the learning rate is linearly scaled up to LR.",
                    )
                    .default_value("10000"),
            )
            .arg(
                Arg::new(WEIGHT_DECAY)
                    .long("weight-decay")
                    .value_name("D")
                    .help("Weight decay (L2 penalty).")
                    .default_value("0.0"),
            );

        SummaryOption::add_to_app(app)
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let config = matches.get_one::<String>(CONFIG).unwrap().into();
        let pretrained_model = matches
            .get_one::<String>(PRETRAINED_MODEL)
            .map(ToOwned::to_owned)
            .unwrap();
        let train_data = matches
            .get_one::<String>(TRAIN_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let validation_data = matches
            .get_one::<String>(VALIDATION_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let batch_size = matches
            .get_one::<String>(BATCH_SIZE)
            .unwrap()
            .parse()
            .context("Cannot parse batch size")?;
        let continue_finetune = matches.get_flag(CONTINUE);
        let device = match matches.get_one::<String>("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .context(format!("Cannot parse GPU number ({})", gpu))?,
            ),
            None => Device::Cpu,
        };
        let finetune_embeds = matches.get_flag(FINETUNE_EMBEDS);
        let initial_lr_classifier = matches
            .get_one::<String>(INITIAL_LR_CLASSIFIER)
            .unwrap()
            .parse()
            .context("Cannot parse initial classifier learning rate")?;
        let initial_lr_encoder = matches
            .get_one::<String>(INITIAL_LR_ENCODER)
            .unwrap()
            .parse()
            .context("Cannot parse initial encoder learning rate")?;
        let label_smoothing = matches
            .get_one::<String>(LABEL_SMOOTHING)
            .map(|v| {
                v.parse()
                    .context(format!("Cannot parse label smoothing probability: {}", v))
            })
            .transpose()?;
        let mixed_precision = matches.get_flag(MIXED_PRECISION);
        let summary_writer = SummaryOption::parse(matches)?;
        let max_len = matches
            .get_one::<String>(MAX_LEN)
            .map(|v| {
                v.parse()
                    .context(format!("Cannot parse maximum sentence length: {}", v))
            })
            .transpose()?
            .map(SequenceLength::Pieces)
            .unwrap_or(SequenceLength::Unbounded);

        let keep_best_epochs = matches
            .get_one::<String>(KEEP_BEST_EPOCHS)
            .map(|n| {
                n.parse()
                    .context("Cannot parse number of best steps to keep")
            })
            .transpose()?;
        if keep_best_epochs == Some(0) {
            bail!("Refusing to keep zero epochs")
        }

        let lr_decay_rate = matches
            .get_one::<String>(LR_DECAY_RATE)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay rate")?;
        let lr_patience = matches
            .get_one::<String>(LR_PATIENCE)
            .unwrap()
            .parse()
            .context("Cannot parse learning rate patience")?;
        let lr_scale = matches
            .get_one::<String>(LR_SCALE)
            .unwrap()
            .parse()
            .context("Cannot parse learning rate scale")?;
        let patience = matches
            .get_one::<String>(PATIENCE)
            .unwrap()
            .parse()
            .context("Cannot parse patience")?;
        let saver = BestEpochSaver::new("", keep_best_epochs);
        let warmup_steps = matches
            .get_one::<String>(WARMUP)
            .unwrap()
            .parse()
            .context("Cannot parse warmup")?;
        let weight_decay = matches
            .get_one::<String>(WEIGHT_DECAY)
            .unwrap()
            .parse()
            .context("Cannot parse weight decay")?;

        Ok(FinetuneApp {
            batch_size,
            config,
            continue_finetune,
            device,
            finetune_embeds,
            max_len,
            label_smoothing,
            mixed_precision,
            summary_writer,
            lr_schedule: LrSchedule {
                initial_lr_encoder,
                initial_lr_classifier,
                lr_decay_rate,
                lr_scale,
                lr_patience,
                warmup_steps,
            },
            patience,
            pretrained_model,
            saver,
            train_data,
            validation_data,
            weight_decay,
        })
    }

    fn run(&self) -> Result<()> {
        let model = if self.continue_finetune {
            Model::load_from(
                &self.config,
                &self.pretrained_model,
                self.device,
                false,
                false,
                Self::build_parameter_group_fun(),
            )?
        } else {
            Model::load_from(
                &self.config,
                &self.pretrained_model,
                self.device,
                false,
                true,
                Self::build_parameter_group_fun(),
            )?
        };

        let mut train_file = File::open(&self.train_data)
            .context(format!("Cannot open train data file: {}", self.train_data))?;
        let mut validation_file = File::open(&self.validation_data).context(format!(
            "Cannot open validation data file: {}",
            self.validation_data
        ))?;

        let mut saver = self.saver.clone();
        let mut grad_scaler = self.build_optimizer(&model.vs)?;

        let mut lr_schedules = self.lr_schedules();

        let mut last_acc = 0.0;
        let mut best_acc = 0.0;
        let mut best_epoch = 0;

        let mut global_step = 1;

        for epoch in 0.. {
            log::info!("Epoch {}", epoch);

            let _ = lr_schedules
                .classifier
                .compute_epoch_learning_rate(epoch, last_acc);
            let _ = lr_schedules
                .encoder
                .compute_epoch_learning_rate(epoch, last_acc);

            self.run_epoch(
                model.biaffine_encoder.as_ref(),
                &model.encoders,
                &*model.tokenizer,
                &model.model,
                &mut train_file,
                Some(&mut grad_scaler),
                &mut lr_schedules,
                &mut global_step,
                epoch,
            )
            .context("Cannot run train epoch")?;

            last_acc = self
                .run_epoch(
                    model.biaffine_encoder.as_ref(),
                    &model.encoders,
                    &*model.tokenizer,
                    &model.model,
                    &mut validation_file,
                    None as Option<&mut GradScaler<nn::Optimizer>>,
                    &mut lr_schedules,
                    &mut global_step,
                    epoch,
                )
                .context("Cannot run valdidation epoch")?;

            if last_acc > best_acc {
                best_epoch = epoch;
                best_acc = last_acc;
            }

            saver
                .save(&model.vs, CompletedUnit::Epoch(last_acc))
                .context("Error saving model")?;

            let epoch_status = if best_epoch == epoch { "🎉" } else { "" };
            log::info!(
                "Epoch {} (validation): acc: {:.4}, best epoch: {}, best acc: {:.4} {}",
                epoch,
                last_acc,
                best_epoch,
                best_acc,
                epoch_status
            );

            self.summary_writer
                .write_scalar("acc:validation,avg", global_step as i64, last_acc)?;

            if epoch - best_epoch == self.patience {
                log::info!(
                    "Lost my patience! Best epoch: {} with accuracy: {:.4}",
                    best_epoch,
                    best_acc
                );
                break;
            }
        }

        Ok(())
    }
}

impl SyntaxDotTrainApp for FinetuneApp {
    fn mixed_precision(&self) -> bool {
        self.mixed_precision
    }

    fn weight_decay(&self) -> f64 {
        self.weight_decay
    }
}
