use std::borrow::Borrow;
use std::collections::HashMap;

use syntaxdot_tch_ext::PathExt;
use syntaxdot_transformers::models::layer_output::LayerOutput;
use syntaxdot_transformers::scalar_weighting::{
    ScalarWeightClassifier, ScalarWeightClassifierConfig,
};
use tch::{Kind, Tensor};

use crate::config::PretrainConfig;
use crate::encoders::Encoders;
use crate::model::bert::PretrainBertConfig;

/// A set of sequence classifiers.
///
/// This data type stores a set of scalar weight-based sequence classifiers,
/// and implements common options for them, such as computing the loss and
/// top-k labels.
#[derive(Debug)]
pub struct SequenceClassifiers {
    classifiers: HashMap<String, ScalarWeightClassifier>,
}

impl SequenceClassifiers {
    /// Create a set of sequence classifiers.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        pretrain_config: &PretrainConfig,
        n_layers: i64,
        encoders: &Encoders,
    ) -> Self {
        let vs = vs.borrow();

        let bert_config = pretrain_config.bert_config();

        let classifiers = encoders
            .iter()
            .map(|encoder| {
                (
                    encoder.name().to_owned(),
                    ScalarWeightClassifier::new(
                        vs.sub("classifiers")
                            .sub(format!("{}_classifier", encoder.name())),
                        &ScalarWeightClassifierConfig {
                            dropout_prob: bert_config.hidden_dropout_prob,
                            hidden_size: bert_config.hidden_size,
                            input_size: bert_config.hidden_size,
                            layer_dropout_prob: 0.1,
                            layer_norm_eps: bert_config.layer_norm_eps,
                            n_layers,
                            n_labels: encoder.encoder().len() as i64,
                        },
                    ),
                )
            })
            .collect();

        SequenceClassifiers { classifiers }
    }

    /// Perform a forward pass of sequence classifiers.
    pub fn forward_t(&self, layers: &[LayerOutput], train: bool) -> HashMap<String, Tensor> {
        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                (encoder_name.to_string(), classifier.logits(layers, train))
            })
            .collect()
    }

    /// Compute the loss of each sequence classifier.
    ///
    /// This method computes the loss of each sequence classifier. Each
    /// classifier performs predictions using the given layer outputs,
    /// masking the tokens in `token_mask`. Then the of the predicted
    /// labels and `targets` is computed.
    ///
    /// If `label_smoothing` is enabled, a the given amount of probability
    /// mass of `targets` is redistributed among other classes than the
    /// targets.
    ///
    /// If `include_continuations` is set to `true`, the loss is also
    /// computed over continuation pieces.
    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        &self,
        layers: &[LayerOutput],
        attention_mask: &Tensor,
        token_mask: &Tensor,
        targets: &HashMap<String, Tensor>,
        label_smoothing: Option<f64>,
        train: bool,
        include_continuations: bool,
    ) -> SequenceClassifiersLoss {
        let token_mask = token_mask.to_kind(Kind::Float);
        let token_mask_sum = token_mask.sum(Kind::Float);

        let mut encoder_losses = HashMap::with_capacity(self.classifiers.len());
        let mut encoder_accuracies = HashMap::with_capacity(self.classifiers.len());
        for (encoder_name, classifier) in &self.classifiers {
            let (loss, correct) =
                classifier.losses(&layers, &targets[encoder_name], label_smoothing, train);
            let loss = if include_continuations {
                (loss * attention_mask).sum(Kind::Float) / &attention_mask.sum(Kind::Float)
            } else {
                (loss * &token_mask).sum(Kind::Float) / &token_mask_sum
            };
            let acc = (correct * &token_mask).sum(Kind::Float) / &token_mask_sum;

            encoder_losses.insert(encoder_name.clone(), loss);
            encoder_accuracies.insert(encoder_name.clone(), acc);
        }

        let summed_loss = encoder_losses.values().fold(
            Tensor::zeros(&[], (Kind::Float, layers[0].output().device())),
            |summed_loss, loss| summed_loss + loss,
        );

        SequenceClassifiersLoss {
            summed_loss,
            encoder_losses,
            encoder_accuracies,
        }
    }

    /// Predict for each classifier the top-K labels and their probabilities.
    ///
    /// This method computes the top-k labels and their probabilities for
    /// each sequence classifier, given the output of each layer. The function
    /// returns a mapping for the classifier name to `(probabilities, labels)`.
    pub fn top_k(&self, layers: &[LayerOutput], k: usize) -> HashMap<String, TopK> {
        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                let (probs, mut labels) = classifier
                    .forward(&layers, false)
                    // Exclude first two classes (padding and continuation).
                    .slice(-1, 2, -1, 1)
                    .topk(k as i64, -1, true, true);

                // Fix label offsets.
                labels += 2;

                (encoder_name.to_string(), TopK { labels, probs })
            })
            .collect()
    }
}

pub struct SequenceClassifiersLoss {
    pub summed_loss: Tensor,
    pub encoder_losses: HashMap<String, Tensor>,
    pub encoder_accuracies: HashMap<String, Tensor>,
}

/// The top-K results for a classifier.
pub struct TopK {
    /// Labels.
    pub labels: Tensor,

    /// Probabilities.
    pub probs: Tensor,
}
