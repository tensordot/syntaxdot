use std::borrow::Borrow;
use std::collections::HashMap;

use syntaxdot_transformers::scalar_weighting::{
    ScalarWeightClassifier, ScalarWeightClassifierConfig,
};
use syntaxdot_transformers::traits::LayerOutput;
use tch::nn::Path;
use tch::{Kind, Tensor};

use crate::config::PretrainConfig;
use crate::encoders::Encoders;
use crate::model::bert::PretrainBertConfig;

#[derive(Debug)]
pub struct SequenceClassifiers {
    classifiers: HashMap<String, ScalarWeightClassifier>,
}

impl SequenceClassifiers {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
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

    pub fn forward_t(&self, layers: &[impl LayerOutput], train: bool) -> HashMap<String, Tensor> {
        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                (encoder_name.to_string(), classifier.logits(layers, train))
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        &self,
        layers: &[impl LayerOutput],
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
            Tensor::zeros(&[], (Kind::Float, layers[0].layer_output().device())),
            |summed_loss, loss| summed_loss + loss,
        );

        SequenceClassifiersLoss {
            summed_loss,
            encoder_losses,
            encoder_accuracies,
        }
    }

    pub fn top_k(&self, layers: &[impl LayerOutput]) -> HashMap<String, (Tensor, Tensor)> {
        self.classifiers
            .iter()
            .map(|(encoder_name, classifier)| {
                let (probs, mut labels) = classifier
                    .forward(&layers, false)
                    // Exclude first two classes (padding and continuation).
                    .slice(-1, 2, -1, 1)
                    .topk(3, -1, true, true);

                // Fix label offsets.
                labels += 2;

                (
                    encoder_name.to_string(),
                    // XXX: make k configurable
                    (probs, labels),
                )
            })
            .collect()
    }
}

pub struct SequenceClassifiersLoss {
    pub summed_loss: Tensor,
    pub encoder_losses: HashMap<String, Tensor>,
    pub encoder_accuracies: HashMap<String, Tensor>,
}
