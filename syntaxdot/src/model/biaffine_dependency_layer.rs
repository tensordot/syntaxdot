use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use syntaxdot_transformers::layers::{
    PairwiseBilinear, PairwiseBilinearConfig, VariationalDropout,
};
use syntaxdot_transformers::loss::CrossEntropyLoss;
use syntaxdot_transformers::models::LayerOutput;
use syntaxdot_transformers::scalar_weighting::ScalarWeight;
use tch::nn::{Init, Linear, Module, ModuleT};
use tch::{Kind, Reduction, Tensor};

use crate::config::{BiaffineParserConfig, PretrainConfig};
use crate::model::bert::PretrainBertConfig;
use crate::tensor::BiaffineTensors;

/// Accuracy of a biaffine parsing layer.
#[derive(Debug)]
pub struct BiaffineAccuracy {
    /// Unlabeled attachment score.
    pub uas: Tensor,

    /// Labeled attachment score.
    pub las: Tensor,
}

/// Loss of a biaffine parsing layer.
#[derive(Debug)]
pub struct BiaffineLoss {
    /// Greedy decoding accuracy.
    pub acc: BiaffineAccuracy,

    /// Head prediction loss.
    pub head_loss: Tensor,

    /// Relation prediction loss.
    pub relation_loss: Tensor,
}

/// Logits of a biaffine parsing layer.
#[derive(Debug)]
pub struct BiaffineScoreLogits {
    /// Head score logits.
    ///
    /// This tensor of shape `[batch_size, seq_len, seq_len]` contains for every
    /// sentence in the batch the scores for each head, given a dependency. For
    /// instance `[s, d, h]` is the score for dependent *b* being attached to
    /// head *h* in sentence *s*.
    pub head_score_logits: Tensor,

    /// This tensor of shape `[batch_size, seq_len, seq_len, n_relations]` contains
    /// dependency relation logits. For instance, if in sentence  *s*, *h* is the head
    /// of *d*, then `[s, d, h]` is the a of the corresponding relation logits.
    pub relation_score_logits: Tensor,
}

/// Biaffine layer for dependency parsing.
#[derive(Debug)]
pub struct BiaffineDependencyLayer {
    scalar_weight: ScalarWeight,

    arc_dependent: Linear,
    arc_head: Linear,
    label_dependent: Linear,
    label_head: Linear,

    bilinear_arc: PairwiseBilinear,
    bilinear_label: PairwiseBilinear,

    dropout: VariationalDropout,
    n_relations: i64,
}

impl BiaffineDependencyLayer {
    /// Construct a new biaffine dependency layer.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        pretrain_config: &PretrainConfig,
        biaffine_config: &BiaffineParserConfig,
        n_relations: i64,
    ) -> Self {
        let bert_config = pretrain_config.bert_config();

        let vs = vs.borrow() / "biaffine";
        let vs = vs.borrow();

        let scalar_weight = ScalarWeight::new(
            vs,
            bert_config.num_hidden_layers,
            bert_config.hidden_dropout_prob,
        );

        let arc_dependent = Self::affine(
            vs / "arc_dependent",
            bert_config.hidden_size,
            biaffine_config.head.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        );

        let arc_head = Self::affine(
            vs / "arc_head",
            bert_config.hidden_size,
            biaffine_config.head.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        );

        let label_dependent = Self::affine(
            vs / "label_dependent",
            bert_config.hidden_size,
            biaffine_config.relation.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        );

        let label_head = Self::affine(
            vs / "label_head",
            bert_config.hidden_size,
            biaffine_config.relation.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        );

        let bilinear_arc = PairwiseBilinear::new(
            vs / "bilinear_arc",
            &PairwiseBilinearConfig {
                bias_u: biaffine_config.head.head_bias,
                bias_v: biaffine_config.head.dependent_bias,
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.head.dims as i64,
                out_features: 1,
            },
        );

        let bilinear_label = PairwiseBilinear::new(
            vs / "bilinear_label",
            &PairwiseBilinearConfig {
                bias_u: biaffine_config.relation.head_bias,
                bias_v: biaffine_config.relation.dependent_bias,
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.relation.dims as i64,
                out_features: n_relations,
            },
        );

        let dropout = VariationalDropout::new(bert_config.hidden_dropout_prob);

        BiaffineDependencyLayer {
            scalar_weight,

            arc_dependent,
            arc_head,
            label_dependent,
            label_head,

            bilinear_arc,
            bilinear_label,

            dropout,
            n_relations,
        }
    }

    fn affine<'a>(
        vs: impl Borrow<PathExt<'a>>,
        in_features: i64,
        out_features: i64,
        initializer_range: f64,
        weight_name: &str,
        bias_name: &str,
    ) -> Linear {
        let vs = vs.borrow();

        Linear {
            ws: vs.var(
                weight_name,
                &[out_features, in_features],
                Init::Randn {
                    mean: 0.,
                    stdev: initializer_range,
                },
            ),
            bs: vs.var(bias_name, &[out_features], Init::Const(0.)),
        }
    }

    /// Apply the biaffine dependency layer.
    ///
    /// The required arguments are:
    ///
    /// * `layers`: encoder output.
    /// * `token_mask`: mask of tokens with shape `[batch_size, seq_len]`.
    /// * `train`: should be `true` when the layer is used in backprop, or `false` otherwise.
    ///
    /// Returns the unnormalized head and label probabilities (logits).
    pub fn forward(
        &self,
        layers: &[LayerOutput],
        token_mask: &Tensor,
        train: bool,
    ) -> BiaffineScoreLogits {
        // Mask for non-tokens (continuations pieces and padding). But do not mask BOS/ROOT
        // as a possible head for each token.
        let logits_mask: Tensor = (1.0 - token_mask.to_kind(Kind::Float)) * -10_000.;
        let _ = logits_mask.slice(1, 0, 1, 1).fill_(0);

        // Get weighted hidden representation.
        let hidden = self.scalar_weight.forward(layers, train);

        // Compute dependent/head arc representations of each token.
        let arc_dependent = self
            .dropout
            .forward_t(&self.arc_dependent.forward(&hidden).gelu(), train);
        let arc_head = self
            .dropout
            .forward_t(&self.arc_head.forward(&hidden).gelu(), train);

        // From these representations, compute the arc score matrix.
        let mut head_score_logits = self.bilinear_arc.forward(&arc_head, &arc_dependent);
        head_score_logits += logits_mask.unsqueeze(1);

        // Compute dependent/head label representations of each token.
        let label_dependent = self
            .dropout
            .forward_t(&self.label_dependent.forward(&hidden).gelu(), train);
        let label_head = self
            .dropout
            .forward_t(&self.label_head.forward(&hidden).gelu(), train);

        // From from these representations, compute the label score matrix.
        let mut relation_score_logits = self.bilinear_label.forward(&label_head, &label_dependent);
        relation_score_logits += logits_mask.unsqueeze(1).unsqueeze(-1);

        BiaffineScoreLogits {
            head_score_logits,
            relation_score_logits,
        }
    }

    /// Compute the biaffine layer loss
    ///
    /// The required arguments are:
    ///
    /// * `layers`: encoder output.
    /// * `token_mask`: mask of tokens with shape `[batch_size, seq_len]`.
    /// * `targets`: the gold-standard dependency heads and dependency relations.
    /// * `label_smoothing`: label smoothing for dependency relations, the given probability
    ///   is distributed among incorrect labels.
    /// * `train`: should be `true` when the layer is used in backprop, or `false` otherwise.
    ///
    /// Returns the loss and greedy decoding LAS/UAS.
    pub fn loss(
        &self,
        layers: &[LayerOutput],
        token_mask: &Tensor,
        targets: &BiaffineTensors<Tensor>,
        label_smoothing: Option<f64>,
        train: bool,
    ) -> BiaffineLoss {
        let biaffine_logits = self.forward(layers, token_mask, train);

        let (batch_size, seq_len) = targets.heads.size2().unwrap();

        // Compute head loss
        let head_logits = biaffine_logits.head_score_logits.view_(&[-1, seq_len]);
        let head_targets = &targets.heads.view_(&[-1]);
        let head_loss = CrossEntropyLoss::new(
            -1,
            // We do not apply label smoothing (yet). It would be strange,
            // since the probabilities of incorrect heads would change with
            // sequence lengths. Also, this would require additional work,
            // since we have to ensure that inactive tokens do not get a
            // probability.
            None,
            Reduction::Mean,
        )
        .forward(&head_logits, &head_targets);

        // Get the logits for the correct heads.
        let label_score_logits = biaffine_logits
            .relation_score_logits
            .gather(
                2,
                &targets
                    .heads
                    // -1 is used for non-token elements, we do not really care
                    // what is selected in these cases, since they won't be used
                    // in the loss.
                    .abs()
                    .view([batch_size, seq_len, 1, 1])
                    .expand(&[-1, -1, 1, self.n_relations], true),
                false,
            )
            .squeeze1(2)
            .view_(&[-1, self.n_relations]);
        let relation_targets = targets.relations.view_(&[-1]);
        let relation_loss = CrossEntropyLoss::new(-1, label_smoothing, Reduction::Mean)
            .forward(&label_score_logits, &relation_targets);

        // Compute greedy decoding accuracy.
        let (uas, las) =
            tch::no_grad(|| Self::greedy_decode_accuracy(&biaffine_logits, targets, &token_mask));

        BiaffineLoss {
            acc: BiaffineAccuracy { uas, las },
            head_loss,
            relation_loss,
        }
    }

    /// Greedily decode the head/relation score tensors and return the LAS/UAS.
    fn greedy_decode_accuracy(
        biaffine_score_logits: &BiaffineScoreLogits,
        targets: &BiaffineTensors<Tensor>,
        token_mask: &Tensor,
    ) -> (Tensor, Tensor) {
        let (batch_size, seq_len) = token_mask.size2().unwrap();

        let token_mask = token_mask.to_kind(Kind::Float);
        let token_mask_sum = token_mask.sum(Kind::Float);

        let head_predicted = biaffine_score_logits.head_score_logits.argmax(-1, false);
        let head_correct = head_predicted.eq1(&targets.heads);

        let relations_predicted = biaffine_score_logits
            .relation_score_logits
            .argmax(-1, false)
            .gather(2, &head_predicted.unsqueeze(-1), false)
            .squeeze();
        let relations_correct = relations_predicted
            .eq1(&targets.relations)
            .view_(&[batch_size, seq_len]);

        let head_and_relations_correct = head_correct.logical_and(&relations_correct);

        let uas = (head_correct * &token_mask).sum(Kind::Float) / &token_mask_sum;
        let las = (head_and_relations_correct * &token_mask).sum(Kind::Float) / &token_mask_sum;

        (uas, las)
    }
}
