//! Scalar weighting of transformer layers.

use std::borrow::Borrow;

use syntaxdot_tch_ext::PathExt;
use tch::nn::{Init, Linear, Module};
use tch::{Kind, Reduction, Tensor};

use crate::cow::CowTensor;
use crate::layers::{Dropout, LayerNorm};
use crate::loss::CrossEntropyLoss;
use crate::models::LayerOutput;
use crate::module::{FallibleModule, FallibleModuleT};
use crate::TransformerError;

/// Non-linear ReLU layer with layer normalization and dropout.
#[derive(Debug)]
struct NonLinearWithLayerNorm {
    layer_norm: LayerNorm,
    linear: Linear,
    dropout: Dropout,
}

impl NonLinearWithLayerNorm {
    fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        in_size: i64,
        out_size: i64,
        dropout: f64,
        layer_norm_eps: f64,
    ) -> Result<NonLinearWithLayerNorm, TransformerError> {
        let vs = vs.borrow();

        Ok(NonLinearWithLayerNorm {
            dropout: Dropout::new(dropout),
            layer_norm: LayerNorm::new(vs / "layer_norm", vec![out_size], layer_norm_eps, true),
            linear: Linear {
                ws: vs.var("weight", &[out_size, in_size], Init::KaimingUniform)?,
                bs: Some(vs.var("bias", &[out_size], Init::Const(0.))?),
            },
        })
    }
}

impl FallibleModuleT for NonLinearWithLayerNorm {
    type Error = TransformerError;

    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        let mut hidden = self.linear.forward(input).relu();
        hidden = self.layer_norm.forward(&hidden)?;
        self.dropout.forward_t(&hidden, train)
    }
}

/// Layer that performs a scalar weighting of layers.
///
/// Following Peters et al., 2018 and Kondratyuk & Straka, 2019, this
/// layer applies scalar weighting:
///
/// *e = c ∑_i[ h_i · softmax(w)_i ]*
#[derive(Debug)]
pub struct ScalarWeight {
    /// Layer dropout probability.
    layer_dropout_prob: f64,

    /// Layer-wise weights.
    layer_weights: Tensor,

    /// Scalar weight.
    scale: Tensor,
}

impl ScalarWeight {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        n_layers: i64,
        layer_dropout_prob: f64,
    ) -> Result<Self, TransformerError> {
        assert!(
            n_layers > 0,
            "Number of layers ({}) should be larger than 0",
            n_layers
        );

        assert!(
            (0.0..1.0).contains(&layer_dropout_prob),
            "Layer dropout should be in [0,1), was: {}",
            layer_dropout_prob
        );

        let vs = vs.borrow();

        Ok(ScalarWeight {
            layer_dropout_prob,
            layer_weights: vs.var("layer_weights", &[n_layers], Init::Const(0.))?,
            scale: vs.var("scale", &[], Init::Const(1.))?,
        })
    }

    pub fn forward(&self, layers: &[LayerOutput], train: bool) -> Result<Tensor, TransformerError> {
        assert_eq!(
            self.layer_weights.size()[0],
            layers.len() as i64,
            "Expected {} layers, got {}",
            self.layer_weights.size()[0],
            layers.len()
        );

        let layers = layers.iter().map(LayerOutput::output).collect::<Vec<_>>();

        // Each layer has shape:
        // [batch_size, sequence_len, layer_size],
        //
        // stack the layers to get a single tensor of shape:
        // [batch_size, sequence_len, n_layers, layer_size]
        let layers = Tensor::f_stack(&layers, 2)?;

        let layer_weights = if train {
            let dropout_mask = Tensor::f_empty_like(&self.layer_weights)?
                .f_fill_(1.0 - self.layer_dropout_prob)?
                .f_bernoulli()?;
            let softmax_mask = (Tensor::from(1.0).f_sub(&dropout_mask.to_kind(Kind::Float))?)
                .f_mul_scalar(-10_000.)?;
            CowTensor::Owned(self.layer_weights.f_add(&softmax_mask)?)
        } else {
            CowTensor::Borrowed(&self.layer_weights)
        };

        // Convert the layer weights into a probability distribution and
        // expand dimensions to get shape [1, 1, n_layers, 1].
        let layer_weights = layer_weights
            .f_softmax(-1, Kind::Float)?
            .f_unsqueeze(0)?
            .f_unsqueeze(0)?
            .f_unsqueeze(-1)?;

        let weighted_layers = layers.f_mul(&layer_weights)?;

        // Sum across all layers and scale.
        Ok(weighted_layers
            .f_sum_dim_intlist(&[-2], false, Kind::Float)?
            .f_mul(&self.scale)?)
    }
}

/// A classifier that uses scalar weighting of layers.
///
/// See Peters et al., 2018 and Kondratyuk & Straka, 2019.
#[derive(Debug)]
pub struct ScalarWeightClassifier {
    dropout: Dropout,
    scalar_weight: ScalarWeight,
    linear: Linear,
    non_linear: NonLinearWithLayerNorm,
}

impl ScalarWeightClassifier {
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        config: &ScalarWeightClassifierConfig,
    ) -> Result<ScalarWeightClassifier, TransformerError> {
        assert!(
            config.n_labels > 0,
            "The number of labels should be larger than 0",
        );

        assert!(
            config.input_size > 0,
            "The input size should be larger than 0",
        );

        assert!(
            config.hidden_size > 0,
            "The hidden size should be larger than 0",
        );

        let vs = vs.borrow();

        let ws = vs.var(
            "weight",
            &[config.n_labels, config.hidden_size],
            Init::KaimingUniform,
        )?;
        let bs = vs.var("bias", &[config.n_labels], Init::Const(0.))?;

        let non_linear = NonLinearWithLayerNorm::new(
            vs / "nonlinear",
            config.input_size,
            config.hidden_size,
            config.dropout_prob,
            config.layer_norm_eps,
        )?;

        Ok(ScalarWeightClassifier {
            dropout: Dropout::new(config.dropout_prob),
            linear: Linear { ws, bs: Some(bs) },
            non_linear,
            scalar_weight: ScalarWeight::new(
                vs / "scalar_weight",
                config.n_layers,
                config.layer_dropout_prob,
            )?,
        })
    }

    pub fn forward(&self, layers: &[LayerOutput], train: bool) -> Result<Tensor, TransformerError> {
        let logits = self.logits(layers, train)?;
        Ok(logits.f_softmax(-1, Kind::Float)?)
    }

    pub fn logits(&self, layers: &[LayerOutput], train: bool) -> Result<Tensor, TransformerError> {
        let mut features = self.scalar_weight.forward(layers, train)?;

        features = self.dropout.forward_t(&features, train)?;

        features = self.non_linear.forward_t(&features, train)?;

        Ok(self.linear.forward(&features))
    }

    /// Compute the losses and correctly predicted labels of the given targets.
    ///
    /// `targets` should be of the shape `[batch_size, seq_len]`.
    pub fn losses(
        &self,
        layers: &[LayerOutput],
        targets: &Tensor,
        label_smoothing: Option<f64>,
        train: bool,
    ) -> Result<(Tensor, Tensor), TransformerError> {
        assert_eq!(
            targets.dim(),
            2,
            "Targets shoul have dimensionality 2, had {}",
            targets.dim()
        );

        let (batch_size, seq_len) = targets.size2()?;

        let n_labels = self.linear.ws.size()[0];

        let logits = self
            .logits(layers, train)?
            .f_view([batch_size * seq_len, n_labels])?;
        let targets = targets.f_view([batch_size * seq_len])?;

        let predicted = logits.f_argmax(-1, false)?;

        let losses = CrossEntropyLoss::new(-1, label_smoothing, Reduction::None)
            .forward(&logits, &targets, None)?
            .f_view([batch_size, seq_len])?;

        Ok((
            losses,
            predicted
                .f_eq_tensor(&targets)?
                .f_view([batch_size, seq_len])?,
        ))
    }
}

/// Configuration for the scalar weight classifier.
pub struct ScalarWeightClassifierConfig {
    /// Size of the hidden layer.
    pub hidden_size: i64,

    /// Size of the input to the classification layer.
    pub input_size: i64,

    /// Number of layers to weigh.
    pub n_layers: i64,

    /// Number of labels.
    pub n_labels: i64,

    /// The probability of excluding a layer from scalar weighting.
    pub layer_dropout_prob: f64,

    /// Hidden layer dropout probability.
    pub dropout_prob: f64,

    /// Layer norm epsilon.
    pub layer_norm_eps: f64,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use syntaxdot_tch_ext::RootExt;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};
    use test_case::test_case;

    use super::{ScalarWeightClassifier, ScalarWeightClassifierConfig};
    use crate::models::{HiddenLayer, LayerOutput};

    fn varstore_variables(vs: &VarStore) -> BTreeSet<String> {
        vs.variables()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<BTreeSet<_>>()
    }

    #[test_case(Device::Cpu)]
    #[cfg_attr(cuda_test, test_case(Device::Cuda(0)))]
    fn scalar_weight_classifier_shapes_forward_works(device: Device) {
        let vs = VarStore::new(device);

        let classifier = ScalarWeightClassifier::new(
            vs.root_ext(|_| 0),
            &ScalarWeightClassifierConfig {
                hidden_size: 10,
                input_size: 8,
                n_labels: 5,
                n_layers: 2,
                dropout_prob: 0.1,
                layer_dropout_prob: 0.1,
                layer_norm_eps: 0.01,
            },
        )
        .unwrap();

        let layer1 = LayerOutput::EncoderWithAttention(HiddenLayer {
            attention: Tensor::zeros(&[1, 3, 2], (Kind::Float, device)),
            output: Tensor::zeros(&[1, 3, 8], (Kind::Float, device)),
        });
        let layer2 = LayerOutput::EncoderWithAttention(HiddenLayer {
            attention: Tensor::zeros(&[1, 3, 2], (Kind::Float, device)),
            output: Tensor::zeros(&[1, 3, 8], (Kind::Float, device)),
        });

        // Perform a forward pass to check that all shapes align.
        let results = classifier.forward(&[layer1, layer2], false).unwrap();

        assert_eq!(results.size(), &[1, 3, 5]);
    }

    #[test_case(Device::Cpu)]
    #[cfg_attr(cuda_test, test_case(Device::Cuda(0)))]
    fn scalar_weight_classifier_names(device: Device) {
        let vs = VarStore::new(device);

        let _classifier = ScalarWeightClassifier::new(
            vs.root_ext(|_| 0),
            &ScalarWeightClassifierConfig {
                hidden_size: 10,
                input_size: 8,
                n_labels: 5,
                n_layers: 2,
                dropout_prob: 0.1,
                layer_dropout_prob: 0.1,
                layer_norm_eps: 0.01,
            },
        );

        assert_eq!(
            varstore_variables(&vs),
            BTreeSet::from_iter(vec![
                "bias".to_string(),
                "weight".to_string(),
                "nonlinear.bias".to_string(),
                "nonlinear.weight".to_string(),
                "nonlinear.layer_norm.bias".to_string(),
                "nonlinear.layer_norm.weight".to_string(),
                "scalar_weight.layer_weights".to_string(),
                "scalar_weight.scale".to_string()
            ])
        )
    }
}
