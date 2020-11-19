use std::cell::RefCell;
use std::collections::btree_map::{BTreeMap, Entry};
use std::fs::File;
use std::io::{BufReader, Read, Seek};

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ordered_float::NotNan;
use syntaxdot::config::Config;
use syntaxdot::dataset::{ConlluDataSet, DataSet, SequenceLength};
use syntaxdot::encoders::Encoders;
use syntaxdot::error::SyntaxDotError;
use syntaxdot::lr::{ExponentialDecay, LearningRateSchedule};
use syntaxdot::model::bert::{BertModel, FreezeLayers};
use syntaxdot::optimizers::{GradScaler, Optimizer};
use syntaxdot::tensor::Tensors;
use syntaxdot::util::seq_len_to_mask;
use syntaxdot_tch_ext::RootExt;
use syntaxdot_tokenizers::Tokenize;
use syntaxdot_transformers::models::layer_output::LayerOutput;
use tch::nn::VarStore;
use tch::{self, Device, Kind, Reduction, Tensor};

use crate::io::{load_config, load_pretrain_config, load_tokenizer, Model};
use crate::progress::ReadProgress;
use crate::summary::{SummaryOption, SummaryWriter};
use crate::traits::{
    ParameterGroup, SyntaxDotApp, SyntaxDotOption, SyntaxDotTrainApp, DEFAULT_CLAP_SETTINGS,
};
use crate::util::{autocast_or_preserve, count_conllu_sentences};

const ATTENTION_LOSS: &str = "ATTENTION_LOSS";
const BATCH_SIZE: &str = "BATCH_SIZE";
const EPOCHS: &str = "EPOCHS";
const EVAL_STEPS: &str = "EVAL_STEPS";
const TEACHER_CONFIG: &str = "TEACHER_CONFIG";
const STUDENT_CONFIG: &str = "STUDENT_CONFIG";
const GPU: &str = "GPU";
const HARD_LOSS: &str = "HARD_LOSS";
const INITIAL_LR_CLASSIFIER: &str = "INITIAL_LR_CLASSIFIER";
const INITIAL_LR_ENCODER: &str = "INITIAL_LR_ENCODER";
const LR_DECAY_RATE: &str = "LR_DECAY_RATE";
const LR_DECAY_STEPS: &str = "LR_DECAY_STEPS";
const MAX_LEN: &str = "MAX_LEN";
const MIXED_PRECISION: &str = "MIXED_PRECISION";
const STEPS: &str = "N_STEPS";
const TRAIN_DATA: &str = "TRAIN_DATA";
const VALIDATION_DATA: &str = "VALIDATION_DATA";
const WARMUP: &str = "WARMUP";
const WEIGHT_DECAY: &str = "WEIGHT_DECAY";

struct DistillLoss {
    pub loss: Tensor,
    pub attention_loss: Tensor,
    pub hard_loss: Tensor,
    pub soft_loss: Tensor,
}

pub struct DistillApp {
    attention_loss: bool,
    batch_size: usize,
    device: Device,
    eval_steps: usize,
    hard_loss: bool,
    max_len: Option<SequenceLength>,
    mixed_precision: bool,
    lr_schedules: RefCell<LearningRateSchedules>,
    student_config: String,
    summary_writer: Box<dyn SummaryWriter>,
    teacher_config: String,
    train_data: String,
    train_duration: TrainDuration,
    validation_data: String,
    weight_decay: f64,
}

pub struct LearningRateSchedules {
    pub classifier: ExponentialDecay,
    pub encoder: ExponentialDecay,
}

struct StudentModel {
    inner: BertModel,
    tokenizer: Box<dyn Tokenize>,
    vs: VarStore,
}

impl DistillApp {
    /// Compute the attention loss based on the output of two encoders.
    ///
    /// The attention loss is the mean squared error of the teacher and student
    /// attentions.
    fn attention_loss(
        &self,
        teacher_layer_outputs: &[LayerOutput],
        student_layer_outputs: &[LayerOutput],
    ) -> Result<Tensor, SyntaxDotError> {
        // Only apply attention loss to layers with attention.
        let teacher_attentions = teacher_layer_outputs
            .iter()
            .filter_map(|l| l.attention())
            .collect::<Vec<_>>();
        let student_attentions = student_layer_outputs
            .iter()
            .filter_map(|l| l.attention())
            .collect::<Vec<_>>();

        let teacher_attention = Tensor::stack(&teacher_attentions, 0);
        let student_attention = Tensor::stack(&student_attentions, 0);

        if student_attention.size() != teacher_attention.size() {
            return Err(SyntaxDotError::IllegalConfigurationError(format!(
                "Cannot compute attention loss: teacher ({:?}) and student ({:?}) have different sequence lengths.",
                teacher_attention
                    .size()
                    .last()
                    .expect("Teacher attention is not a tensor"),
                student_attention
                    .size()
                    .last()
                    .expect("Student attention is not a tensor")
            )));
        }

        // The attention matrix uses logits. Tokens are masked by giving them very
        // negative values. Remove such tokens by removing extreme negative values.
        // The threshold is the same as that used by TinyBERT.
        let zeros = Tensor::zeros_like(&teacher_attention);
        let teacher_attention = teacher_attention.where1(&teacher_attention.lt(-1e2), &zeros);
        let student_attention = student_attention.where1(&student_attention.lt(-1e2), &zeros);

        Ok(student_attention.mse_loss(&teacher_attention, Reduction::Mean))
    }

    fn distill_model(
        &self,
        grad_scaler: &mut GradScaler<impl Optimizer>,
        teacher: &Model,
        student: &StudentModel,
        teacher_train_file: &File,
        student_train_file: &File,
        validation_file: &mut File,
    ) -> Result<()> {
        let mut best_step = 0;
        let mut best_acc = 0.0;

        let mut global_step = 0;

        let n_steps = self
            .train_duration
            .to_steps(&teacher_train_file, self.batch_size)
            .context("Cannot determine number of training steps")?;

        let train_progress = ProgressBar::new(n_steps as u64);
        train_progress.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% train {msg}",
        ));

        while global_step < n_steps - 1 {
            let mut teacher_train_dataset = Self::open_dataset(&teacher_train_file)?;
            let mut student_train_dataset = Self::open_dataset(&student_train_file)?;

            let teacher_train_batches = teacher_train_dataset.batches(
                &*teacher.tokenizer,
                None,
                self.batch_size,
                self.max_len,
                None,
            )?;

            let student_train_batches = student_train_dataset.batches(
                &*student.tokenizer,
                None,
                self.batch_size,
                self.max_len,
                None,
            )?;

            for (teacher_steps, student_steps) in teacher_train_batches
                .chunks(self.eval_steps)
                .into_iter()
                .zip(student_train_batches.chunks(self.eval_steps).into_iter())
            {
                self.train_steps(
                    &train_progress,
                    teacher_steps,
                    student_steps,
                    &mut global_step,
                    grad_scaler,
                    &teacher.model,
                    &student.inner,
                )?;

                let acc = self.validation_epoch(
                    &teacher.encoders,
                    &*student.tokenizer,
                    &student.inner,
                    validation_file,
                    global_step,
                )?;

                self.summary_writer
                    .write_scalar("acc:validation,avg", global_step as i64, acc)?;

                if acc > best_acc {
                    best_step = global_step;
                    best_acc = acc;

                    student
                        .vs
                        .save(format!("distill-step-{}", global_step))
                        .context(format!(
                            "Cannot save variable store for step {}",
                            global_step
                        ))?;
                }

                let step_status = if best_step == global_step { "🎉" } else { "" };

                eprintln!(
                    "Step {} (validation): acc: {:.4}, best step: {}, best acc: {:.4} {}\n",
                    global_step, acc, best_step, best_acc, step_status
                );

                if global_step >= n_steps - 1 {
                    break;
                }
            }
        }

        Ok(())
    }

    fn student_loss(
        &self,
        teacher: &BertModel,
        student: &BertModel,
        teacher_batch: Tensors,
        student_batch: Tensors,
    ) -> Result<DistillLoss, SyntaxDotError> {
        // Compute masks.
        let teacher_attention_mask =
            seq_len_to_mask(&teacher_batch.seq_lens, teacher_batch.inputs.size()[1])
                .to_device(self.device);
        let teacher_token_mask = teacher_batch
            .token_mask
            .to_kind(Kind::Bool)
            .to_device(self.device);

        let teacher_layer_outputs = teacher.encode(
            &teacher_batch.inputs.to_device(self.device),
            &teacher_attention_mask,
            false,
            FreezeLayers {
                embeddings: true,
                encoder: true,
                classifiers: true,
            },
        );
        let teacher_logits = teacher.logits_from_encoding(&teacher_layer_outputs, false);

        let student_attention_mask =
            seq_len_to_mask(&student_batch.seq_lens, student_batch.inputs.size()[1])
                .to_device(self.device);
        let student_token_mask = student_batch
            .token_mask
            .to_kind(Kind::Bool)
            .to_device(self.device);

        autocast_or_preserve(self.mixed_precision, || {
            let student_layer_outputs = student.encode(
                &student_batch.inputs.to_device(self.device),
                &student_attention_mask,
                true,
                FreezeLayers {
                    embeddings: false,
                    encoder: false,
                    classifiers: false,
                },
            );
            let student_logits = student.logits_from_encoding(&student_layer_outputs, true);

            let mut soft_loss = Tensor::zeros(&[], (Kind::Float, self.device));
            let mut hard_loss = Tensor::zeros(&[], (Kind::Float, self.device));

            for (encoder_name, teacher_logits) in teacher_logits {
                let n_labels = teacher_logits.size()[2];

                // Select the outputs for the relevant time steps.
                let student_logits = student_logits[&encoder_name]
                    .masked_select(&student_token_mask.unsqueeze(-1))
                    .reshape(&[-1, n_labels]);
                let teacher_logits = teacher_logits
                    .masked_select(&teacher_token_mask.unsqueeze(-1))
                    .reshape(&[-1, n_labels]);

                // Compute the soft loss.
                let teacher_probs = teacher_logits.softmax(-1, Kind::Float);
                let student_logprobs = student_logits.log_softmax(-1, Kind::Float);
                let soft_losses = -(&teacher_probs * &student_logprobs);
                soft_loss += soft_losses
                    .sum1(&[-1], false, Kind::Float)
                    .mean(Kind::Float);

                if self.hard_loss {
                    let teacher_predictions = teacher_logits.argmax(-1, false);

                    hard_loss += student_logprobs.g_nll_loss::<&Tensor>(
                        &teacher_predictions,
                        None,
                        Reduction::Mean,
                        -100,
                    );
                }
            }

            let attention_loss = if self.attention_loss {
                self.attention_loss(&teacher_layer_outputs, &student_layer_outputs)?
            } else {
                Tensor::zeros(&[], (Kind::Float, self.device))
            };

            Ok(DistillLoss {
                loss: &hard_loss + &soft_loss + &attention_loss,
                attention_loss,
                hard_loss,
                soft_loss,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_steps(
        &self,
        progress: &ProgressBar,
        teacher_batches: impl Iterator<Item = Result<Tensors, SyntaxDotError>>,
        student_batches: impl Iterator<Item = Result<Tensors, SyntaxDotError>>,
        global_step: &mut usize,
        grad_scaler: &mut GradScaler<impl Optimizer>,
        teacher: &BertModel,
        student: &BertModel,
    ) -> Result<()> {
        for (teacher_batch, student_batch) in teacher_batches.zip(student_batches) {
            let teacher_batch = teacher_batch.context("Cannot read teacher batch")?;
            let student_batch = student_batch.context("Cannot read student batch")?;

            let distill_loss = self.student_loss(teacher, student, teacher_batch, student_batch)?;

            let lr_classifier = self
                .lr_schedules
                .borrow_mut()
                .classifier
                .compute_step_learning_rate(*global_step);
            let lr_encoder = self
                .lr_schedules
                .borrow_mut()
                .encoder
                .compute_step_learning_rate(*global_step);

            let optimizer = grad_scaler.optimizer_mut();

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

            grad_scaler.backward_step(&distill_loss.loss);

            self.summary_writer.write_scalar(
                "gradient_scale",
                *global_step as i64,
                grad_scaler.current_scale(),
            )?;

            progress.set_message(&format!(
                "step: {} | lr enc: {:+.1e}, class: {:+.1e} | loss hard: {:+.1e}, soft: {:+.1e}, attention: {:+.1e}",
                global_step,
                lr_encoder,
                lr_classifier,
                f32::from(distill_loss.hard_loss),
                f32::from(distill_loss.soft_loss),
                f32::from(distill_loss.attention_loss)
            ));
            progress.inc(1);

            *global_step += 1;
        }

        Ok(())
    }

    fn open_dataset(file: &File) -> Result<ConlluDataSet<impl Read + Seek>> {
        let read = BufReader::new(
            file.try_clone()
                .context("Cannot open data set for reading")?,
        );
        Ok(ConlluDataSet::new(read))
    }

    fn fresh_student(
        &self,
        student_config: &Config,
        teacher: &Model,
        parameter_group_fun: impl Fn(&str) -> usize + 'static,
    ) -> Result<StudentModel> {
        let pretrain_config = load_pretrain_config(student_config)?;

        let vs = VarStore::new(self.device);

        let inner = BertModel::new(
            vs.root_ext(parameter_group_fun),
            &pretrain_config,
            &teacher.encoders,
            0.1,
            student_config.model.position_embeddings.clone(),
        )
        .context("Cannot construct fresh student model")?;

        let tokenizer = load_tokenizer(&student_config)?;

        Ok(StudentModel {
            inner,
            tokenizer,
            vs,
        })
    }

    pub fn create_lr_schedules(
        initial_lr_classifier: NotNan<f32>,
        initial_lr_encoder: NotNan<f32>,
        lr_decay_rate: NotNan<f32>,
        lr_decay_steps: usize,
        warmup_steps: usize,
    ) -> LearningRateSchedules {
        let classifier = ExponentialDecay::new(
            initial_lr_classifier.into_inner(),
            lr_decay_rate.into_inner(),
            lr_decay_steps,
            false,
            warmup_steps,
        );

        let mut encoder = classifier.clone();
        encoder.set_initial_lr(initial_lr_encoder.into_inner());

        LearningRateSchedules {
            classifier,
            encoder,
        }
    }

    fn validation_epoch(
        &self,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        global_step: usize,
    ) -> Result<f32> {
        let read_progress = ReadProgress::new(file).context("Cannot create progress bar")?;
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% validation {msg}",
        ));

        let mut dataset = ConlluDataSet::new(read_progress);

        let mut encoder_accuracy = BTreeMap::new();
        let mut encoder_loss = BTreeMap::new();

        let mut n_tokens = 0;

        for batch in dataset.batches(
            tokenizer,
            Some(encoders),
            self.batch_size,
            self.max_len,
            None,
        )? {
            let batch = batch?;

            let n_batch_tokens = i64::from(batch.token_mask.sum(Kind::Int64));

            let attention_mask = seq_len_to_mask(&batch.seq_lens, batch.inputs.size()[1]);

            let model_loss = autocast_or_preserve(self.mixed_precision, || {
                model.loss(
                    &batch.inputs.to_device(self.device),
                    &attention_mask.to_device(self.device),
                    &batch.token_mask.to_device(self.device),
                    &batch
                        .labels
                        .expect("Batch without labels.")
                        .into_iter()
                        .map(|(encoder_name, labels)| (encoder_name, labels.to_device(self.device)))
                        .collect(),
                    None,
                    false,
                    FreezeLayers {
                        embeddings: true,
                        encoder: true,
                        classifiers: true,
                    },
                    false,
                )
            });

            n_tokens += n_batch_tokens;

            let scalar_loss: f32 = model_loss.summed_loss.sum(Kind::Float).into();

            for (encoder_name, loss) in model_loss.encoder_losses {
                match encoder_accuracy.entry(encoder_name.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(
                            f32::from(&model_loss.encoder_accuracies[&encoder_name])
                                * n_batch_tokens as f32,
                        );
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() +=
                            f32::from(&model_loss.encoder_accuracies[&encoder_name])
                                * n_batch_tokens as f32;
                    }
                };

                match encoder_loss.entry(encoder_name) {
                    Entry::Vacant(entry) => {
                        entry.insert(f32::from(loss) * n_batch_tokens as f32);
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() += f32::from(loss) * n_batch_tokens as f32
                    }
                };
            }

            progress_bar.set_message(&format!(
                "batch loss: {:.4}, global step: {}",
                scalar_loss, global_step
            ));
        }

        progress_bar.finish();

        eprintln!();
        let mut acc_sum = 0.0;
        for (encoder_name, loss) in encoder_loss {
            let acc = encoder_accuracy[&encoder_name] / n_tokens as f32;
            let loss = loss / n_tokens as f32;

            eprintln!("{} loss: {} accuracy: {:.4}", encoder_name, loss, acc);

            self.summary_writer.write_scalar(
                &format!("loss:validation,layer:{}", &encoder_name),
                global_step as i64,
                loss,
            )?;

            self.summary_writer.write_scalar(
                &format!("acc:validation,layer:{}", &encoder_name),
                global_step as i64,
                acc,
            )?;

            acc_sum += acc;
        }
        eprintln!();

        Ok(acc_sum / encoders.len() as f32)
    }
}

impl SyntaxDotApp for DistillApp {
    fn app() -> App<'static, 'static> {
        let app = App::new("distill")
            .settings(DEFAULT_CLAP_SETTINGS)
            .about("Distill a model")
            .arg(
                Arg::with_name(TEACHER_CONFIG)
                    .help("Teacher configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(STUDENT_CONFIG)
                    .help("Student configuration file")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name(TRAIN_DATA)
                    .help("Training data")
                    .index(3)
                    .required(true),
            )
            .arg(
                Arg::with_name(VALIDATION_DATA)
                    .help("Validation data")
                    .index(4)
                    .required(true),
            )
            .arg(
                Arg::with_name(ATTENTION_LOSS)
                    .long("attention-loss")
                    .help("Add attention score loss"),
            )
            .arg(
                Arg::with_name(BATCH_SIZE)
                    .long("batch-size")
                    .takes_value(true)
                    .help("Batch size")
                    .default_value("32"),
            )
            .arg(
                Arg::with_name(EPOCHS)
                    .long("epochs")
                    .takes_value(true)
                    .value_name("N")
                    .help("Train for N epochs")
                    .default_value("2"),
            )
            .arg(
                Arg::with_name(EVAL_STEPS)
                    .long("eval-steps")
                    .takes_value(true)
                    .value_name("N")
                    .help("Evaluate after N steps, save the model on improvement")
                    .default_value("1000"),
            )
            .arg(
                Arg::with_name(GPU)
                    .long("gpu")
                    .takes_value(true)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::with_name(HARD_LOSS)
                    .long("hard-loss")
                    .help("Add hard loss (predicted label) to the soft loss"),
            )
            .arg(
                Arg::with_name(INITIAL_LR_CLASSIFIER)
                    .long("lr-classifier")
                    .value_name("LR")
                    .help("Initial classifier learning rate")
                    .default_value("1e-3"),
            )
            .arg(
                Arg::with_name(INITIAL_LR_ENCODER)
                    .long("lr-encoder")
                    .value_name("LR")
                    .help("Initial encoder learning rate")
                    .default_value("5e-5"),
            )
            .arg(
                Arg::with_name(MIXED_PRECISION)
                    .long("mixed-precision")
                    .help("Enable automatic mixed-precision"),
            )
            .arg(
                Arg::with_name(LR_DECAY_RATE)
                    .long("lr-decay-rate")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("0.99998"),
            )
            .arg(
                Arg::with_name(LR_DECAY_STEPS)
                    .long("lr-decay-steps")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("10"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(STEPS)
                    .long("steps")
                    .value_name("N")
                    .help("Train for N steps")
                    .takes_value(true)
                    .overrides_with(EPOCHS),
            )
            .arg(
                Arg::with_name(WARMUP)
                    .long("warmup")
                    .value_name("N")
                    .help(
                        "For the first N timesteps, the learning rate is linearly scaled up to LR.",
                    )
                    .default_value("2000"),
            )
            .arg(
                Arg::with_name(WEIGHT_DECAY)
                    .long("weight-decay")
                    .value_name("D")
                    .help("Weight decay (L2 penalty).")
                    .default_value("0.0"),
            );

        SummaryOption::add_to_app(app)
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let teacher_config = matches.value_of(TEACHER_CONFIG).unwrap().into();
        let student_config = matches.value_of(STUDENT_CONFIG).unwrap().into();
        let train_data = matches.value_of(TRAIN_DATA).map(ToOwned::to_owned).unwrap();
        let validation_data = matches
            .value_of(VALIDATION_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let attention_loss = matches.is_present(ATTENTION_LOSS);
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
        let eval_steps = matches
            .value_of(EVAL_STEPS)
            .unwrap()
            .parse()
            .context("Cannot parse number of batches after which to save")?;
        let hard_loss = matches.is_present(HARD_LOSS);
        let initial_lr_classifier = matches
            .value_of(INITIAL_LR_CLASSIFIER)
            .unwrap()
            .parse()
            .context("Cannot parse initial classifier learning rate")?;
        let initial_lr_encoder = matches
            .value_of(INITIAL_LR_ENCODER)
            .unwrap()
            .parse()
            .context("Cannot parse initial encoder learning rate")?;
        let summary_writer = SummaryOption::parse(matches)?;
        let lr_decay_rate = matches
            .value_of(LR_DECAY_RATE)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay rate")?;
        let lr_decay_steps = matches
            .value_of(LR_DECAY_STEPS)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay steps")?;
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().context("Cannot parse maximum sentence length"))
            .transpose()?
            .map(SequenceLength::Tokens);
        let mixed_precision = matches.is_present(MIXED_PRECISION);
        let warmup_steps = matches
            .value_of(WARMUP)
            .unwrap()
            .parse()
            .context("Cannot parse warmup")?;
        let weight_decay = matches
            .value_of(WEIGHT_DECAY)
            .unwrap()
            .parse()
            .context("Cannot parse weight decay")?;

        // If steps is present, it overrides epochs.
        let train_duration = if let Some(steps) = matches.value_of(STEPS) {
            let steps = steps
                .parse()
                .context("Cannot parse the number of training steps")?;
            TrainDuration::Steps(steps)
        } else {
            let epochs = matches
                .value_of(EPOCHS)
                .unwrap()
                .parse()
                .context("Cannot parse number of training epochs")?;
            TrainDuration::Epochs(epochs)
        };

        Ok(DistillApp {
            attention_loss,
            batch_size,
            device,
            eval_steps,
            hard_loss,
            max_len,
            mixed_precision,
            lr_schedules: RefCell::new(Self::create_lr_schedules(
                initial_lr_classifier,
                initial_lr_encoder,
                lr_decay_rate,
                lr_decay_steps,
                warmup_steps,
            )),
            student_config,
            teacher_config,
            summary_writer,
            train_data,
            train_duration,
            validation_data,
            weight_decay,
        })
    }

    fn run(&self) -> Result<()> {
        let student_config = load_config(&self.student_config)?;
        let teacher = Model::load(&self.teacher_config, self.device, true, |_| 0)?;

        let teacher_train_file = File::open(&self.train_data)
            .context(format!("Cannot open train data file: {}", self.train_data))?;
        let student_train_file = File::open(&self.train_data)
            .context(format!("Cannot open train data file: {}", self.train_data))?;
        let mut validation_file = File::open(&self.validation_data).context(format!(
            "Cannot open validation data file: {}",
            self.validation_data
        ))?;

        let student =
            self.fresh_student(&student_config, &teacher, Self::build_parameter_group_fun())?;

        let mut grad_scaler = self.build_optimizer(&student.vs)?;

        self.distill_model(
            &mut grad_scaler,
            &teacher,
            &student,
            &teacher_train_file,
            &student_train_file,
            &mut validation_file,
        )
        .context("Model distillation failed")
    }
}

impl SyntaxDotTrainApp for DistillApp {
    fn mixed_precision(&self) -> bool {
        self.mixed_precision
    }

    fn weight_decay(&self) -> f64 {
        self.weight_decay
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainDuration {
    Epochs(usize),
    Steps(usize),
}

impl TrainDuration {
    fn to_steps(&self, train_file: &File, batch_size: usize) -> Result<usize> {
        use TrainDuration::*;

        match *self {
            Epochs(epochs) => {
                eprintln!("Counting number of steps in an epoch...");
                let read_progress =
                    ReadProgress::new(train_file.try_clone()?).context("Cannot open train file")?;

                let progress_bar = read_progress.progress_bar().clone();
                progress_bar
                    .set_style(ProgressStyle::default_bar().template(
                        "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}%",
                    ));

                let n_sentences = count_conllu_sentences(BufReader::new(read_progress))?;

                progress_bar.finish_and_clear();

                // Compute number of steps of the given batch size.
                let steps_per_epoch = (n_sentences + batch_size - 1) / batch_size;
                eprintln!(
                    "sentences: {}, steps_per epoch: {}, total_steps: {}",
                    n_sentences,
                    steps_per_epoch,
                    epochs * steps_per_epoch
                );
                Ok(epochs * steps_per_epoch)
            }
            Steps(steps) => Ok(steps),
        }
    }
}
