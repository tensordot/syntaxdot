use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use syntaxdot_transformers::layers::{
    PairwiseBilinear, PairwiseBilinearConfig, VariationalDropout,
};
use syntaxdot_transformers::loss::CrossEntropyLoss;
use syntaxdot_transformers::models::LayerOutput;
use syntaxdot_transformers::module::FallibleModuleT;
use syntaxdot_transformers::scalar_weighting::ScalarWeight;
use tch::nn::{Init, Linear, Module};
use tch::{Kind, Reduction, Tensor};

use crate::config::{BiaffineParserConfig, PretrainConfig};
use crate::error::SyntaxDotError;
use crate::model::bert::PretrainBertConfig;
use crate::tensor::{BiaffineTensors, TokenMask};

/// Accuracy of a biaffine parsing layer.
#[derive(Debug)]
pub struct BiaffineAccuracy {
    /// Labeled attachment score.
    pub las: Tensor,

    /// Label score.
    pub ls: Tensor,

    /// Unlabeled attachment score.
    pub uas: Tensor,
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
        n_layers: i64,
        n_relations: i64,
    ) -> Result<BiaffineDependencyLayer, SyntaxDotError> {
        let bert_config = pretrain_config.bert_config();

        let vs = vs.borrow() / "biaffine";
        let vs = vs.borrow();

        let scalar_weight = ScalarWeight::new(vs, n_layers, bert_config.hidden_dropout_prob)?;

        let arc_dependent = Self::affine(
            vs / "arc_dependent",
            bert_config.hidden_size,
            biaffine_config.head.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        )?;

        let arc_head = Self::affine(
            vs / "arc_head",
            bert_config.hidden_size,
            biaffine_config.head.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        )?;

        let label_dependent = Self::affine(
            vs / "label_dependent",
            bert_config.hidden_size,
            biaffine_config.relation.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        )?;

        let label_head = Self::affine(
            vs / "label_head",
            bert_config.hidden_size,
            biaffine_config.relation.dims as i64,
            bert_config.initializer_range,
            "weight",
            "bias",
        )?;

        let bilinear_arc = PairwiseBilinear::new(
            vs / "bilinear_arc",
            &PairwiseBilinearConfig {
                bias_u: biaffine_config.head.head_bias,
                bias_v: biaffine_config.head.dependent_bias,
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.head.dims as i64,
                out_features: 1,
            },
        )?;

        let bilinear_label = PairwiseBilinear::new(
            vs / "bilinear_label",
            &PairwiseBilinearConfig {
                bias_u: biaffine_config.relation.head_bias,
                bias_v: biaffine_config.relation.dependent_bias,
                initializer_range: bert_config.initializer_range,
                in_features: biaffine_config.relation.dims as i64,
                out_features: n_relations,
            },
        )?;

        let dropout = VariationalDropout::new(bert_config.hidden_dropout_prob);

        Ok(BiaffineDependencyLayer {
            scalar_weight,

            arc_dependent,
            arc_head,
            label_dependent,
            label_head,

            bilinear_arc,
            bilinear_label,

            dropout,
            n_relations,
        })
    }

    fn affine<'a>(
        vs: impl Borrow<PathExt<'a>>,
        in_features: i64,
        out_features: i64,
        initializer_range: f64,
        weight_name: &str,
        bias_name: &str,
    ) -> Result<Linear, SyntaxDotError> {
        let vs = vs.borrow();

        Ok(Linear {
            ws: vs.var(
                weight_name,
                &[out_features, in_features],
                Init::Randn {
                    mean: 0.,
                    stdev: initializer_range,
                },
            )?,
            bs: vs.var(bias_name, &[out_features], Init::Const(0.))?,
        })
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
        token_mask: &TokenMask,
        remove_root: bool,
        train: bool,
    ) -> Result<BiaffineScoreLogits, SyntaxDotError> {
        let token_mask_with_root = token_mask.with_root()?;

        // Mask padding. But do not mask BOS/ROOT as a possible head for each token.
        let logits_mask: Tensor = Tensor::from(1.0)
            .f_sub(&token_mask_with_root.to_kind(Kind::Float))?
            .f_mul1(-10_000.)?;
        let _ = logits_mask.f_slice(1, 0, 1, 1)?.f_fill_(0)?;

        // Get weighted hidden representation.
        let hidden = self.scalar_weight.forward(layers, train)?;

        // Compute dependent/head arc representations of each token.
        let arc_dependent = self
            .dropout
            .forward_t(&self.arc_dependent.forward(&hidden).gelu(), train)?;
        let arc_head = self
            .dropout
            .forward_t(&self.arc_head.forward(&hidden).gelu(), train)?;

        // From these representations, compute the arc score matrix.
        let head_score_logits = self
            .bilinear_arc
            .forward(&arc_head, &arc_dependent)?
            // Mask padding logits.
            .f_add_(&logits_mask.f_unsqueeze(1)?)?;

        // Compute dependent/head label representations of each token.
        let label_dependent = self
            .dropout
            .forward_t(&self.label_dependent.forward(&hidden).f_gelu()?, train)?;
        let label_head = self
            .dropout
            .forward_t(&self.label_head.forward(&hidden).f_gelu()?, train)?;

        // From from these representations, compute the label score matrix.
        let relation_score_logits = self
            .bilinear_label
            .forward(&label_head, &label_dependent)?
            // Mask padding logits.
            .f_add_(&logits_mask.f_unsqueeze(1)?.f_unsqueeze(-1)?)?;

        if remove_root {
            Ok(BiaffineScoreLogits {
                head_score_logits: head_score_logits.f_slice(1, 1, i64::MAX, 1)?,
                relation_score_logits: relation_score_logits.f_slice(1, 1, i64::MAX, 1)?,
            })
        } else {
            Ok(BiaffineScoreLogits {
                head_score_logits,
                relation_score_logits,
            })
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
        token_mask: &TokenMask,
        targets: &BiaffineTensors<Tensor>,
        label_smoothing: Option<f64>,
        train: bool,
    ) -> Result<BiaffineLoss, SyntaxDotError> {
        assert_eq!(
            targets.heads.dim(),
            2,
            "Head targets should have dimensionality 2, had {}",
            targets.heads.dim()
        );
        assert_eq!(
            targets.relations.dim(),
            2,
            "Relation targets should have dimensionality 2, had {}",
            targets.relations.dim()
        );
        assert_eq!(
            token_mask.dim(),
            2,
            "Token mask should have dimensionality 2, had {}",
            token_mask.dim()
        );

        let biaffine_logits = self.forward(layers, &token_mask, true, train)?;

        let (batch_size, seq_len) = targets.heads.size2()?;

        let token_mask_with_root = token_mask.with_root()?;

        // Compute head loss
        let head_logits = biaffine_logits
            .head_score_logits
            // Last dimension is ROOT + all tokens as head candidates.
            .f_reshape(&[-1, seq_len + 1])?;
        let head_targets = &targets.heads.f_view_(&[-1])?;
        let head_loss = CrossEntropyLoss::new(-1, label_smoothing, Reduction::Mean).forward(
            &head_logits,
            &head_targets,
            Some(
                &token_mask_with_root
                    // [batch_size, seq_len + 1] -> [batch_size, 1, seq_len + 1]
                    .f_unsqueeze(1)?
                    // [batch_size, 1, seq_len + 1] -> [batch_size, seq_len, seq_len + 1].
                    .f_expand(&[-1, seq_len, -1], true)?
                    // [batch_size, seq_len, seq_len + 1] -> [batch_size * seq_len, seq_len + 1]
                    .f_reshape(&[-1, seq_len + 1])?,
            ),
        )?;

        // Get the logits for the correct heads.
        let label_score_logits = biaffine_logits
            .relation_score_logits
            .f_gather(
                2,
                &targets
                    .heads
                    // -1 is used for non-token elements, we do not really care
                    // what is selected in these cases, since they won't be used
                    // in the loss.
                    .f_abs()?
                    .f_view([batch_size, seq_len, 1, 1])?
                    .f_expand(&[-1, -1, 1, self.n_relations], true)?,
                false,
            )?
            .f_squeeze1(2)?
            .f_view_(&[-1, self.n_relations])?;

        let relation_targets = targets.relations.f_view_(&[-1])?;
        let relation_loss = CrossEntropyLoss::new(-1, label_smoothing, Reduction::Mean).forward(
            &label_score_logits,
            &relation_targets,
            None,
        )?;

        // Compute greedy decoding accuracy.
        let acc =
            tch::no_grad(|| Self::greedy_decode_accuracy(&biaffine_logits, targets, &token_mask))?;

        Ok(BiaffineLoss {
            acc,
            head_loss,
            relation_loss,
        })
    }

    /// Greedily decode the head/relation score tensors and return the LAS/UAS.
    fn greedy_decode_accuracy(
        biaffine_score_logits: &BiaffineScoreLogits,
        targets: &BiaffineTensors<Tensor>,
        token_mask: &TokenMask,
    ) -> Result<BiaffineAccuracy, SyntaxDotError> {
        let (batch_size, seq_len) = token_mask.size2()?;

        let head_predicted = biaffine_score_logits
            .head_score_logits
            .f_argmax(-1, false)?;
        let head_correct = head_predicted.f_eq1(&targets.heads)?;

        let relations_predicted = biaffine_score_logits
            .relation_score_logits
            .f_argmax(-1, false)?
            .f_gather(2, &head_predicted.f_unsqueeze(-1)?, false)?
            .f_squeeze()?;
        let relations_correct = relations_predicted
            .f_eq1(&targets.relations)?
            .f_view_(&[batch_size, seq_len])?;

        let head_and_relations_correct = head_correct.f_logical_and(&relations_correct)?;

        let las = head_and_relations_correct
            .f_masked_select(&token_mask)?
            .f_mean(Kind::Float)?;
        let ls = relations_correct
            .f_masked_select(&token_mask)?
            .f_mean(Kind::Float)?;
        let uas = head_correct
            .f_masked_select(&token_mask)?
            .f_mean(Kind::Float)?;

        Ok(BiaffineAccuracy { las, ls, uas })
    }
}
