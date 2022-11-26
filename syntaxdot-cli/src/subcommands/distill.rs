use std::cell::RefCell;
use std::collections::btree_map::{BTreeMap, Entry};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ordered_float::NotNan;
use syntaxdot::config::{Config, PretrainConfig};
use syntaxdot::dataset::{
    BatchedTensors, ConlluDataSet, DataSet, PlainTextDataSet, SentenceIterTools, SequenceLength,
};
use syntaxdot::encoders::Encoders;
use syntaxdot::error::SyntaxDotError;
use syntaxdot::lr::{ExponentialDecay, LearningRateSchedule};
use syntaxdot::model::bert::{BertModel, FreezeLayers, PretrainBertConfig};
use syntaxdot::model::biaffine_dependency_layer::BiaffineScoreLogits;
use syntaxdot::optimizers::{GradScaler, Optimizer};
use syntaxdot::tensor::{Tensors, TokenMask};
use syntaxdot_encoders::dependency::ImmutableDependencyEncoder;
use syntaxdot_tch_ext::tensor::SumDim;
use syntaxdot_tch_ext::RootExt;
use syntaxdot_tokenizers::Tokenize;
use syntaxdot_transformers::models::LayerOutput;
use tch::nn::{Init, VarStore};
use tch::{self, Device, Kind, Reduction, Tensor};

use crate::io::{load_config, load_pretrain_config, load_tokenizer, Model};
use crate::progress::ReadProgress;
use crate::summary::{ScalarWriter, SummaryOption};
use crate::traits::{ParameterGroup, SyntaxDotApp, SyntaxDotOption, SyntaxDotTrainApp};
use crate::util::{autocast_or_preserve, count_sentences};

const ATTENTION_LOSS: &str = "ATTENTION_LOSS";
const BATCH_SIZE: &str = "BATCH_SIZE";
const EPOCHS: &str = "EPOCHS";
const EVAL_STEPS: &str = "EVAL_STEPS";
const HIDDEN_LOSS: &str = "HIDDEN_LOSS";
const TEACHER_CONFIG: &str = "TEACHER_CONFIG";
const STUDENT_CONFIG: &str = "STUDENT_CONFIG";
const GPU: &str = "GPU";
const INITIAL_LR_CLASSIFIER: &str = "INITIAL_LR_CLASSIFIER";
const INITIAL_LR_ENCODER: &str = "INITIAL_LR_ENCODER";
const KEEP_BEST_STEPS: &str = "KEEP_BEST_STEPS";
const LR_DECAY_RATE: &str = "LR_DECAY_RATE";
const LR_DECAY_STEPS: &str = "LR_DECAY_STEPS";
const MAX_LEN: &str = "MAX_LEN";
const MIXED_PRECISION: &str = "MIXED_PRECISION";
const STEPS: &str = "N_STEPS";
const TRAIN_DATA: &str = "TRAIN_DATA";
const VALIDATION_DATA: &str = "VALIDATION_DATA";
const WARMUP: &str = "WARMUP";
const WEIGHT_DECAY: &str = "WEIGHT_DECAY";

/// Auxiliary model parameters used during distillation.
///
/// These are additional student model parameters that are only
/// necessary for some distillation options.
struct AuxiliaryParameters {
    /// Hidden layer lappings.
    hidden_mappings: Option<Vec<HiddenMapping>>,
}

/// Hidden layer mapping.
struct HiddenMapping {
    /// Teacher hidden layer.
    teacher_layer: usize,

    /// Student hidden layer.
    student_layer: usize,

    /// Transform student hidden layer space to teacher.
    mapping: Tensor,
}

struct BiaffineEpochStats {
    // Labeled attachment score.
    las: f32,

    // Label score.
    ls: f32,

    // Unlabeled attachment score.
    uas: f32,

    head_loss: f32,

    relation_loss: f32,
}

struct DistillLoss {
    pub loss: Tensor,
    pub attention_loss: Tensor,
    pub hidden_loss: Tensor,
    pub soft_loss: Tensor,
}

pub struct DistillApp {
    attention_loss: bool,
    batch_size: usize,
    device: Device,
    eval_steps: usize,
    hidden_loss: Option<Vec<(usize, usize)>>,
    keep_best_steps: Option<usize>,
    max_len: SequenceLength,
    mixed_precision: bool,
    lr_schedules: RefCell<LearningRateSchedules>,
    student_config: String,
    summary_writer: Box<dyn ScalarWriter>,
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
    pretrain_config: PretrainConfig,
    tokenizer: Box<dyn Tokenize>,
    vs: VarStore,
}

struct EpochStats {
    biaffine: Option<BiaffineEpochStats>,
    encoder_accuracy: BTreeMap<String, f32>,
    encoder_loss: BTreeMap<String, f32>,
    n_tokens: i64,
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
        let teacher_attention =
            teacher_attention.where_scalarother(&teacher_attention.lt(-1e2), 0.);
        let student_attention =
            student_attention.where_scalarother(&student_attention.lt(-1e2), 0.);

        Ok(student_attention.mse_loss(&teacher_attention, Reduction::Mean))
    }

    fn biaffine_loss(
        teacher_logits: &BiaffineScoreLogits,
        student_logits: &BiaffineScoreLogits,
        token_mask: &TokenMask,
    ) -> Result<Tensor> {
        // Compute teacher probabilities.
        let teacher_head_probs = teacher_logits
            .head_score_logits
            .f_softmax(-1, Kind::Float)?;

        let probs_mask = token_mask
            .with_root()?
            .unsqueeze(1)
            .logical_and(&token_mask.unsqueeze(-1));
        let teacher_head_probs = teacher_head_probs.masked_select(&probs_mask);

        // Compute student log scores.
        let student_head_logprobs = student_logits
            .head_score_logits
            .f_log_softmax(-1, Kind::Float)?;
        let student_head_logprobs = student_head_logprobs.masked_select(&probs_mask);

        let head_soft_loss =
            (&teacher_head_probs.f_mul(&student_head_logprobs)?.f_neg()?).f_mean(Kind::Float)?;

        // Get teacher relation probabilities for the heads predicted by the teacher.
        let teacher_relation_probs = teacher_logits
            .relation_score_logits
            .f_softmax(-1, Kind::Float)?;

        // Get student relation log probabilities for the heads predicted by the teacher.
        let student_relation_log_probs = student_logits
            .relation_score_logits
            .f_log_softmax(-1, Kind::Float)?;

        let relation_soft_loss = teacher_relation_probs
            .f_neg()?
            .f_mul(&student_relation_log_probs)?
            .f_mean(Kind::Float)?;

        Ok(head_soft_loss.f_add(&relation_soft_loss)?)
    }

    fn create_auxiliary_params(
        &self,
        vs: &VarStore,
        teacher: &PretrainConfig,
        student: &dyn PretrainBertConfig,
    ) -> Result<AuxiliaryParameters> {
        let root = vs.root_ext(|_| ParameterGroup::Classifier as usize);
        let path = root.sub("auxiliary_params");

        // Create hidden layer mappings.
        let hidden_mappings = self
            .hidden_loss
            .as_ref()
            .map(|layer_mapping| {
                layer_mapping
                    .iter()
                    .map(|&(teacher_layer, student_layer)| {
                        let mapping = path.var(
                            &format!("hidden_mapping_{}_{}", teacher_layer, student_layer),
                            &[
                                student.bert_config().hidden_size,
                                teacher.bert_config().hidden_size,
                            ],
                            Init::Randn {
                                mean: 0.0,
                                stdev: student.bert_config().initializer_range,
                            },
                        )?;

                        Ok(HiddenMapping {
                            teacher_layer,
                            student_layer,
                            mapping,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        Ok(AuxiliaryParameters { hidden_mappings })
    }

    #[allow(clippy::too_many_arguments)]
    fn distill_model(
        &self,
        grad_scaler: &mut GradScaler<impl Optimizer>,
        auxiliary_params: &AuxiliaryParameters,
        teacher: &Model,
        student: &StudentModel,
        teacher_train_file: &File,
        student_train_file: &File,
        validation_file: &mut File,
    ) -> Result<()> {
        let mut best_step = 0;
        let mut best_acc = 0.0;

        let mut global_step = 0;

        let mut best_step_paths = self.keep_best_steps.map(VecDeque::with_capacity);

        let n_steps = self
            .train_duration
            .as_steps(teacher_train_file, self.batch_size)
            .context("Cannot determine number of training steps")?;

        let train_progress = ProgressBar::new(n_steps as u64);
        train_progress.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% train {msg}",
        ));

        while global_step < n_steps - 1 {
            let mut teacher_train_dataset = Self::open_dataset(teacher_train_file)?;
            let mut student_train_dataset = Self::open_dataset(student_train_file)?;

            let teacher_train_batches = teacher_train_dataset
                .sentences(&*teacher.tokenizer)?
                .filter_by_len(self.max_len)
                .batched_tensors(None, None, self.batch_size);

            let student_train_batches = student_train_dataset
                .sentences(&*student.tokenizer)?
                .filter_by_len(self.max_len)
                .batched_tensors(None, None, self.batch_size);

            for (teacher_steps, student_steps) in teacher_train_batches
                .chunks(self.eval_steps)
                .into_iter()
                .zip(student_train_batches.chunks(self.eval_steps).into_iter())
            {
                self.train_steps(
                    auxiliary_params,
                    &train_progress,
                    teacher_steps,
                    student_steps,
                    &mut global_step,
                    grad_scaler,
                    &teacher.model,
                    &student.inner,
                )?;

                let acc = self.validation_epoch(
                    teacher.biaffine_encoder.as_ref(),
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

                    let step_path = format!("distill-step-{}", global_step);

                    student.vs.save(&step_path).context(format!(
                        "Cannot save variable store for step {}",
                        global_step
                    ))?;

                    self.cleanup_old_best_steps(&mut best_step_paths, step_path);
                }

                let step_status = if best_step == global_step { "🎉" } else { "" };

                log::info!(
                    "Step {} (validation): acc: {:.4}, best step: {}, best acc: {:.4} {}\n",
                    global_step,
                    acc,
                    best_step,
                    best_acc,
                    step_status
                );

                if global_step >= n_steps - 1 {
                    break;
                }
            }
        }

        Ok(())
    }

    fn cleanup_old_best_steps(
        &self,
        best_step_paths: &mut Option<VecDeque<String>>,
        step_path: String,
    ) {
        if let Some(best_step_paths) = best_step_paths.as_mut() {
            if best_step_paths.len() == self.keep_best_steps.unwrap() {
                let cleanup_step = best_step_paths.pop_front().expect("No steps?");
                if let Err(err) = fs::remove_file(&cleanup_step) {
                    log::error!("Cannot remove step parameters {}: {}", cleanup_step, err);
                }
            }

            best_step_paths.push_back(step_path);
        }
    }

    /// Hidden layer loss.
    fn hidden_loss(
        &self,
        token_mask: &TokenMask,
        mappings: &[HiddenMapping],
        teacher_layer_outputs: &[LayerOutput],
        student_layer_outputs: &[LayerOutput],
    ) -> Result<Tensor> {
        let token_mask = token_mask.with_root()?;

        let mut loss = Tensor::zeros(&[], (Kind::Float, self.device));

        for mapping in mappings {
            let teacher_output = teacher_layer_outputs
                .get(mapping.teacher_layer)
                .ok_or_else(|| anyhow!("Unknown teacher layer: {}", mapping.teacher_layer))?;
            let student_output = student_layer_outputs
                .get(mapping.student_layer)
                .ok_or_else(|| anyhow!("Unknown student layer: {}", mapping.student_layer))?;

            let teacher_hidden = teacher_output
                .output()
                .f_masked_select(&token_mask.f_unsqueeze(-1)?)?;
            let student_hidden = student_output
                .output()
                .f_matmul(&mapping.mapping)?
                .f_masked_select(&token_mask.f_unsqueeze(-1)?)?;

            let _ = loss.f_add_(&student_hidden.f_mse_loss(&teacher_hidden, Reduction::Mean)?)?;
        }

        Ok(loss)
    }

    fn parse_layer_mappings(layers: &str) -> Result<Vec<(usize, usize)>> {
        let mut mappings = Vec::new();

        for layer_mapping in layers.split(',') {
            let mut parts_iter = layer_mapping.split(':');

            let teacher_layer = parts_iter
                .next()
                .ok_or_else(|| anyhow!("No teacher layer in mapping: {}", layer_mapping))?
                .parse()
                .context("Cannot parse teacher layer")?;
            let student_layer = parts_iter
                .next()
                .ok_or_else(|| anyhow!("No student layer in mapping: {}", layer_mapping))?
                .parse()
                .context("Cannot parse student layer")?;

            if parts_iter.next().is_some() {
                bail!(
                    "Layer mapping contains too many components: {}",
                    layer_mapping
                );
            }

            mappings.push((teacher_layer, student_layer));
        }

        Ok(mappings)
    }

    /// Compute loss for sequence encoders.
    fn seq_encoders_loss(
        teacher_encoder_logits: HashMap<String, Tensor>,
        student_encoder_logits: HashMap<String, Tensor>,
        token_mask: &TokenMask,
    ) -> Result<Tensor, SyntaxDotError> {
        let mut loss = Tensor::zeros(&[], (Kind::Float, token_mask.device()));

        for (encoder_name, teacher_logits) in teacher_encoder_logits {
            let n_labels = teacher_logits.size()[2];

            // Select the outputs for the relevant time steps.
            let student_logits = student_encoder_logits[&encoder_name]
                .masked_select(&token_mask.unsqueeze(-1))
                .reshape(&[-1, n_labels]);
            let teacher_logits = teacher_logits
                .masked_select(&token_mask.unsqueeze(-1))
                .reshape(&[-1, n_labels]);

            // Compute the soft loss.
            let teacher_probs = teacher_logits.f_softmax(-1, Kind::Float)?;
            let student_logprobs = student_logits.f_log_softmax(-1, Kind::Float)?;
            let soft_losses = teacher_probs.f_mul(&student_logprobs)?.f_neg()?;
            let _ = loss.f_add_(
                &soft_losses
                    .f_sum_dim(-1, false, Kind::Float)?
                    .f_mean(Kind::Float)?,
            )?;
        }

        Ok(loss)
    }

    fn student_loss(
        &self,
        auxiliary_params: &AuxiliaryParameters,
        teacher: &BertModel,
        student: &BertModel,
        teacher_batch: Tensors,
        student_batch: Tensors,
    ) -> Result<DistillLoss> {
        // Compute masks.
        let teacher_attention_mask = teacher_batch
            .seq_lens
            .attention_mask()?
            .to_device(self.device);

        let teacher_token_spans = teacher_batch.token_spans.to_device(self.device);
        let token_mask = teacher_token_spans.token_mask()?;

        let teacher_layer_outputs = teacher.encode(
            &teacher_batch.inputs.to_device(self.device),
            &teacher_attention_mask,
            &teacher_token_spans,
            false,
            FreezeLayers {
                embeddings: true,
                encoder: true,
                classifiers: true,
            },
        )?;
        let teacher_encoder_logits =
            teacher.encoder_logits_from_encoding(&teacher_layer_outputs, false)?;
        let teacher_biaffine_logits =
            teacher.biaffine_logits_from_encoding(&teacher_layer_outputs, &token_mask, false)?;

        let student_attention_mask = student_batch
            .seq_lens
            .attention_mask()?
            .to_device(self.device);
        let student_token_spans = student_batch.token_spans.to_device(self.device);

        autocast_or_preserve(self.mixed_precision, || {
            let student_layer_outputs = student.encode(
                &student_batch.inputs.to_device(self.device),
                &student_attention_mask,
                &student_token_spans,
                true,
                FreezeLayers {
                    embeddings: false,
                    encoder: false,
                    classifiers: false,
                },
            )?;

            let mut soft_loss = Tensor::zeros(&[], (Kind::Float, self.device));

            // Compute biaffine encoder/decoder loss.
            match (
                teacher_biaffine_logits,
                student.biaffine_logits_from_encoding(
                    &student_layer_outputs,
                    &token_mask,
                    true,
                )?,
            ) {
                (Some(teacher_logits), Some(student_logits)) => {
                    let _ = soft_loss.f_add_(&Self::biaffine_loss(&teacher_logits,
                                                     &student_logits, &token_mask )?)?;
                }
                (None, Some(_)) => bail!("Cannot distill biaffine parsing model from a teacher without biaffine parsing."),
                _ => {}
            }

            // Compute sequence encoder/decoder loss.
            let student_logits =
                student.encoder_logits_from_encoding(&student_layer_outputs, true)?;
            let _ = soft_loss.f_add_(&Self::seq_encoders_loss(
                teacher_encoder_logits,
                student_logits,
                &token_mask,
            )?)?;

            let attention_loss = if self.attention_loss {
                self.attention_loss(&teacher_layer_outputs, &student_layer_outputs)?
            } else {
                Tensor::zeros(&[], (Kind::Float, self.device))
            };

            let hidden_loss = match auxiliary_params.hidden_mappings {
                Some(ref mappings) => self.hidden_loss(
                    &token_mask,
                    mappings,
                    &teacher_layer_outputs,
                    &student_layer_outputs,
                )?,
                None => Tensor::zeros(&[], (Kind::Float, self.device)),
            };

            Ok(DistillLoss {
                loss: soft_loss.f_add(&attention_loss)?.f_add(&hidden_loss)?,
                attention_loss,
                hidden_loss,
                soft_loss,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn train_steps(
        &self,
        auxiliary_params: &AuxiliaryParameters,
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

            let distill_loss = self.student_loss(
                auxiliary_params,
                teacher,
                student,
                teacher_batch,
                student_batch,
            )?;

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

            grad_scaler.backward_step(&distill_loss.loss)?;

            self.summary_writer.write_scalar(
                "gradient_scale",
                *global_step as i64,
                grad_scaler.current_scale(),
            )?;

            progress.set_message(format!(
                "step: {} | lr enc: {:+.1e}, class: {:+.1e} | loss soft: {:+.1e}, attention: {:+.1e}, hidden: {:+.1e}",
                global_step,
                lr_encoder,
                lr_classifier,
                f32::from(distill_loss.soft_loss),
                f32::from(distill_loss.attention_loss),
                f32::from(distill_loss.hidden_loss)
            ));
            progress.inc(1);

            *global_step += 1;
        }

        Ok(())
    }

    fn open_dataset(file: &File) -> Result<PlainTextDataSet<impl BufRead + Seek>> {
        let read = BufReader::new(
            file.try_clone()
                .context("Cannot open data set for reading")?,
        );
        Ok(PlainTextDataSet::new(read))
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
            student_config.biaffine.as_ref(),
            teacher
                .biaffine_encoder
                .as_ref()
                .map(ImmutableDependencyEncoder::n_relations)
                .unwrap_or(0),
            &teacher.encoders,
            student_config.model.pooler,
            0.1,
            student_config.model.position_embeddings.clone(),
        )
        .context("Cannot construct fresh student model")?;

        let tokenizer = load_tokenizer(student_config)?;

        Ok(StudentModel {
            inner,
            pretrain_config,
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
        biaffine_encoder: Option<&ImmutableDependencyEncoder>,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        global_step: usize,
    ) -> Result<f32> {
        let epoch_stats = self.validation_epoch_steps(
            biaffine_encoder,
            encoders,
            tokenizer,
            model,
            file,
            global_step,
        )?;
        self.log_epoch_stats(global_step, epoch_stats)
    }

    fn log_epoch_stats(&self, global_step: usize, epoch_stats: EpochStats) -> Result<f32> {
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
                "loss:validation,biaffine:head",
                global_step as i64,
                biaffine_stats.head_loss / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                "loss:validation,biaffine:relation",
                global_step as i64,
                biaffine_stats.relation_loss / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                "las:validation,biaffine",
                global_step as i64,
                biaffine_stats.las / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                "ls:validation,biaffine",
                global_step as i64,
                biaffine_stats.ls / epoch_stats.n_tokens as f32,
            )?;

            self.summary_writer.write_scalar(
                "uas:validation,biaffine",
                global_step as i64,
                biaffine_stats.uas / epoch_stats.n_tokens as f32,
            )?;
        }

        for (encoder_name, loss) in epoch_stats.encoder_loss {
            let acc = epoch_stats.encoder_accuracy[&encoder_name] / epoch_stats.n_tokens as f32;
            let loss = loss / epoch_stats.n_tokens as f32;

            log::info!("{} loss: {} accuracy: {:.4}", encoder_name, loss, acc);

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

            accs.push(acc);
        }

        Ok(accs.iter().sum::<f32>() / accs.len() as f32)
    }

    fn validation_epoch_steps(
        &self,
        biaffine_encoder: Option<&ImmutableDependencyEncoder>,
        encoders: &Encoders,
        tokenizer: &dyn Tokenize,
        model: &BertModel,
        file: &mut File,
        global_step: usize,
    ) -> Result<EpochStats> {
        let read_progress = ReadProgress::new(file).context("Cannot create progress bar")?;
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% validation {msg}",
        ));

        let mut dataset = ConlluDataSet::new(BufReader::new(read_progress));

        let mut biaffine_las = 0f32;
        let mut biaffine_ls = 0f32;
        let mut biaffine_uas = 0f32;
        let mut biaffine_head_loss = 0f32;
        let mut biaffine_relation_loss = 0f32;
        let mut encoder_accuracy = BTreeMap::new();
        let mut encoder_loss = BTreeMap::new();

        let mut n_tokens = 0;

        for batch in dataset
            .sentences(tokenizer)?
            .filter_by_len(self.max_len)
            .batched_tensors(biaffine_encoder, Some(encoders), self.batch_size)
        {
            let batch = batch?;

            let n_batch_tokens = i64::from(batch.token_spans.token_mask()?.f_sum(Kind::Int64)?);

            let attention_mask = batch.seq_lens.attention_mask()?;

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
                    None,
                    false,
                    FreezeLayers {
                        embeddings: true,
                        encoder: true,
                        classifiers: true,
                    },
                )
            })?;

            n_tokens += n_batch_tokens;

            let scalar_loss: f32 = model_loss
                .seq_classifiers
                .summed_loss
                .f_sum(Kind::Float)?
                .into();

            for (encoder_name, loss) in model_loss.seq_classifiers.encoder_losses {
                match encoder_accuracy.entry(encoder_name.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(
                            f32::from(
                                &model_loss.seq_classifiers.encoder_accuracies[&encoder_name],
                            ) * n_batch_tokens as f32,
                        );
                    }
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() += f32::from(
                            &model_loss.seq_classifiers.encoder_accuracies[&encoder_name],
                        ) * n_batch_tokens as f32;
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

            if let Some(biaffine_loss) = model_loss.biaffine.as_ref() {
                let head_loss = f32::from(&biaffine_loss.head_loss);
                let relation_loss = f32::from(&biaffine_loss.relation_loss);

                biaffine_las += f32::from(&biaffine_loss.acc.las) * n_batch_tokens as f32;
                biaffine_ls += f32::from(&biaffine_loss.acc.ls) * n_batch_tokens as f32;
                biaffine_uas += f32::from(&biaffine_loss.acc.uas) * n_batch_tokens as f32;
                biaffine_head_loss += head_loss * n_batch_tokens as f32;
                biaffine_relation_loss += relation_loss * n_batch_tokens as f32;

                progress_bar.set_message(format!(
                    "seq loss: {:.4}, head loss: {:.4}, rel los: {:.4}, global step: {}",
                    scalar_loss, head_loss, relation_loss, global_step
                ));
            } else {
                progress_bar.set_message(format!(
                    "seq loss: {:.4}, global step: {}",
                    scalar_loss, global_step
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

impl SyntaxDotApp for DistillApp {
    fn app() -> Command {
        let app = Command::new("distill")
            .arg_required_else_help(true)
            .about("Distill a model")
            .arg(
                Arg::new(TEACHER_CONFIG)
                    .help("Teacher configuration file")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::new(STUDENT_CONFIG)
                    .help("Student configuration file")
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
                Arg::new(ATTENTION_LOSS)
                    .long("attention-loss")
                    .action(ArgAction::SetTrue)
                    .help("Add attention score loss"),
            )
            .arg(
                Arg::new(BATCH_SIZE)
                    .long("batch-size")
                    .num_args(1)
                    .help("Batch size")
                    .default_value("32"),
            )
            .arg(
                Arg::new(EPOCHS)
                    .long("epochs")
                    .num_args(1)
                    .value_name("N")
                    .help("Train for N epochs")
                    .default_value("2"),
            )
            .arg(
                Arg::new(EVAL_STEPS)
                    .long("eval-steps")
                    .num_args(1)
                    .value_name("N")
                    .help("Evaluate after N steps, save the model on improvement")
                    .default_value("1000"),
            )
            .arg(
                Arg::new(GPU)
                    .long("gpu")
                    .num_args(1)
                    .help("Use the GPU with the given identifier"),
            )
            .arg(
                Arg::new(HIDDEN_LOSS)
                    .long("hidden-loss")
                    .value_name("MAPPING")
                    .num_args(1)
                    .help("Add hidden representations MSE loss"),
            )
            .arg(
                Arg::new(INITIAL_LR_CLASSIFIER)
                    .long("lr-classifier")
                    .value_name("LR")
                    .help("Initial classifier learning rate")
                    .default_value("5e-5"),
            )
            .arg(
                Arg::new(INITIAL_LR_ENCODER)
                    .long("lr-encoder")
                    .value_name("LR")
                    .help("Initial encoder learning rate")
                    .default_value("5e-5"),
            )
            .arg(
                Arg::new(KEEP_BEST_STEPS)
                    .long("keep-best")
                    .value_name("N")
                    .help("Only keep the N best steps"),
            )
            .arg(
                Arg::new(MIXED_PRECISION)
                    .long("mixed-precision")
                    .action(ArgAction::SetTrue)
                    .help("Enable automatic mixed-precision"),
            )
            .arg(
                Arg::new(LR_DECAY_RATE)
                    .long("lr-decay-rate")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("0.99998"),
            )
            .arg(
                Arg::new(LR_DECAY_STEPS)
                    .long("lr-decay-steps")
                    .value_name("N")
                    .help("Exponential decay rate")
                    .default_value("10"),
            )
            .arg(
                Arg::new(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .num_args(1)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::new(STEPS)
                    .long("steps")
                    .value_name("N")
                    .help("Train for N steps")
                    .num_args(1)
                    .overrides_with(EPOCHS),
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
        let teacher_config = matches.get_one::<String>(TEACHER_CONFIG).unwrap().into();
        let student_config = matches.get_one::<String>(STUDENT_CONFIG).unwrap().into();
        let train_data = matches
            .get_one::<String>(TRAIN_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let validation_data = matches
            .get_one::<String>(VALIDATION_DATA)
            .map(ToOwned::to_owned)
            .unwrap();
        let attention_loss = matches.get_flag(ATTENTION_LOSS);
        let batch_size = matches
            .get_one::<String>(BATCH_SIZE)
            .unwrap()
            .parse()
            .context("Cannot parse batch size")?;
        let device = match matches.get_one::<String>("GPU") {
            Some(gpu) => Device::Cuda(
                gpu.parse()
                    .context(format!("Cannot parse GPU number ({})", gpu))?,
            ),
            None => Device::Cpu,
        };
        let eval_steps = matches
            .get_one::<String>(EVAL_STEPS)
            .unwrap()
            .parse()
            .context("Cannot parse number of batches after which to save")?;
        let hidden_loss = matches
            .get_one::<String>(HIDDEN_LOSS)
            .map(|s| Self::parse_layer_mappings(s.as_str()))
            .transpose()?;
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
        let summary_writer = SummaryOption::parse(matches)?;

        let keep_best_steps = matches
            .get_one::<String>(KEEP_BEST_STEPS)
            .map(|n| {
                n.parse()
                    .context("Cannot parse number of best steps to keep")
            })
            .transpose()?;
        if keep_best_steps == Some(0) {
            bail!("Refusing to keep zero steps")
        }

        let lr_decay_rate = matches
            .get_one::<String>(LR_DECAY_RATE)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay rate")?;
        let lr_decay_steps = matches
            .get_one::<String>(LR_DECAY_STEPS)
            .unwrap()
            .parse()
            .context("Cannot parse exponential decay steps")?;
        let max_len = matches
            .get_one::<String>(MAX_LEN)
            .map(|v| v.parse().context("Cannot parse maximum sentence length"))
            .transpose()?
            .map(SequenceLength::Tokens)
            .unwrap_or(SequenceLength::Unbounded);
        let mixed_precision = matches.get_flag(MIXED_PRECISION);
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

        // If steps is present, it overrides epochs.
        let train_duration = if let Some(steps) = matches.get_one::<String>(STEPS) {
            let steps = steps
                .parse()
                .context("Cannot parse the number of training steps")?;
            TrainDuration::Steps(steps)
        } else {
            let epochs = matches
                .get_one::<String>(EPOCHS)
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
            hidden_loss,
            keep_best_steps,
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
        let teacher = Model::load(&self.teacher_config, self.device, true, false, |_| 0)?;

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

        let auxiliary_params = self.create_auxiliary_params(
            &student.vs,
            &teacher.pretrain_config,
            &student.pretrain_config,
        )?;

        self.distill_model(
            &mut grad_scaler,
            &auxiliary_params,
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
    fn as_steps(&self, train_file: &File, batch_size: usize) -> Result<usize> {
        use TrainDuration::*;

        match *self {
            Epochs(epochs) => {
                log::info!("Counting number of steps in an epoch...");
                let read_progress =
                    ReadProgress::new(train_file.try_clone()?).context("Cannot open train file")?;

                let progress_bar = read_progress.progress_bar().clone();
                progress_bar
                    .set_style(ProgressStyle::default_bar().template(
                        "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}%",
                    ));

                let n_sentences = count_sentences(BufReader::new(read_progress))?;

                progress_bar.finish_and_clear();

                // Compute number of steps of the given batch size.
                let steps_per_epoch = (n_sentences + batch_size - 1) / batch_size;
                log::info!(
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
