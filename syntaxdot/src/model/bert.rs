use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::time::Instant;

use syntaxdot_tch_ext::PathExt;
use syntaxdot_transformers::layers::Dropout;
use syntaxdot_transformers::models::albert::{AlbertConfig, AlbertEmbeddings, AlbertEncoder};
use syntaxdot_transformers::models::bert::{BertConfig, BertEmbeddings, BertEncoder};
use syntaxdot_transformers::models::roberta::RobertaEmbeddings;
use syntaxdot_transformers::models::sinusoidal::SinusoidalEmbeddings;
use syntaxdot_transformers::models::squeeze_albert::SqueezeAlbertEncoder;
use syntaxdot_transformers::models::squeeze_bert::SqueezeBertEncoder;
use syntaxdot_transformers::models::Encoder as _;
use syntaxdot_transformers::models::LayerOutput;
use syntaxdot_transformers::module::FallibleModuleT;
use syntaxdot_transformers::TransformerError;
use tch::{self, Tensor};

use crate::config::{BiaffineParserConfig, PositionEmbeddings, PretrainConfig};
use crate::encoders::Encoders;
use crate::error::SyntaxDotError;
use crate::model::biaffine_dependency_layer::{
    BiaffineDependencyLayer, BiaffineLoss, BiaffineScoreLogits,
};
use crate::model::seq_classifiers::{SequenceClassifiers, SequenceClassifiersLoss, TopK};
use crate::tensor::BiaffineTensors;

pub trait PretrainBertConfig {
    fn bert_config(&self) -> Cow<BertConfig>;
}

impl PretrainBertConfig for PretrainConfig {
    fn bert_config(&self) -> Cow<BertConfig> {
        match self {
            PretrainConfig::Albert(config) => Cow::Owned(config.into()),
            PretrainConfig::Bert(config) => Cow::Borrowed(config),
            PretrainConfig::SqueezeAlbert(config) => Cow::Owned(config.into()),
            PretrainConfig::SqueezeBert(config) => Cow::Owned(config.into()),
            PretrainConfig::XlmRoberta(config) => Cow::Borrowed(config),
        }
    }
}

#[derive(Debug)]
enum BertEmbeddingLayer {
    Albert(AlbertEmbeddings),
    Bert(BertEmbeddings),
    Roberta(RobertaEmbeddings),
    Sinusoidal(SinusoidalEmbeddings),
}

impl BertEmbeddingLayer {
    fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        pretrain_config: &PretrainConfig,
        position_embeddings: PositionEmbeddings,
    ) -> Result<BertEmbeddingLayer, SyntaxDotError> {
        let vs = vs.borrow();

        let embedding_layer = match (pretrain_config, position_embeddings) {
            (PretrainConfig::Albert(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Albert(AlbertEmbeddings::new(vs / "embeddings", config)?)
            }
            (PretrainConfig::Albert(config), PositionEmbeddings::Sinusoidal { normalize }) => {
                let normalize = if normalize { Some(2.) } else { None };
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(
                    vs / "embeddings",
                    config,
                    normalize,
                )?)
            }
            (PretrainConfig::Bert(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Bert(BertEmbeddings::new(vs / "embeddings", config)?)
            }
            (PretrainConfig::Bert(config), PositionEmbeddings::Sinusoidal { normalize }) => {
                let normalize = if normalize { Some(2.) } else { None };
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(
                    vs / "embeddings",
                    config,
                    normalize,
                )?)
            }
            (PretrainConfig::SqueezeAlbert(config), PositionEmbeddings::Model) => {
                let albert_config: AlbertConfig = config.into();
                BertEmbeddingLayer::Albert(AlbertEmbeddings::new(
                    vs / "embeddings",
                    &albert_config,
                )?)
            }
            (
                PretrainConfig::SqueezeAlbert(config),
                PositionEmbeddings::Sinusoidal { normalize },
            ) => {
                let normalize = if normalize { Some(2.) } else { None };
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(
                    vs / "embeddings",
                    config,
                    normalize,
                )?)
            }
            (PretrainConfig::SqueezeBert(config), PositionEmbeddings::Model) => {
                let bert_config: BertConfig = config.into();
                BertEmbeddingLayer::Bert(BertEmbeddings::new(vs / "embeddings", &bert_config)?)
            }
            (PretrainConfig::SqueezeBert(config), PositionEmbeddings::Sinusoidal { normalize }) => {
                let bert_config: BertConfig = config.into();
                let normalize = if normalize { Some(2.) } else { None };
                BertEmbeddingLayer::Sinusoidal(SinusoidalEmbeddings::new(
                    vs / "embeddings",
                    &bert_config,
                    normalize,
                )?)
            }
            (PretrainConfig::XlmRoberta(config), PositionEmbeddings::Model) => {
                BertEmbeddingLayer::Roberta(RobertaEmbeddings::new(vs / "embeddings", config)?)
            }
            (PretrainConfig::XlmRoberta(_), PositionEmbeddings::Sinusoidal { .. }) => {
                unreachable!()
            }
        };

        Ok(embedding_layer)
    }
}

impl FallibleModuleT for BertEmbeddingLayer {
    type Error = SyntaxDotError;

    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor, Self::Error> {
        use BertEmbeddingLayer::*;

        let embeddings = match self {
            Albert(ref embeddings) => embeddings.forward_t(input, train),
            Bert(ref embeddings) => embeddings.forward_t(input, train),
            Roberta(ref embeddings) => embeddings.forward_t(input, train),
            Sinusoidal(ref embeddings) => embeddings.forward_t(input, train),
        }?;

        Ok(embeddings)
    }
}

#[derive(Debug)]
enum Encoder {
    Albert(AlbertEncoder),
    Bert(BertEncoder),
    SqueezeAlbert(SqueezeAlbertEncoder),
    SqueezeBert(SqueezeBertEncoder),
}

impl Encoder {
    fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        pretrain_config: &PretrainConfig,
    ) -> Result<Self, TransformerError> {
        let vs = vs.borrow() / "encoder";

        let encoder = match pretrain_config {
            PretrainConfig::Albert(config) => Encoder::Albert(AlbertEncoder::new(vs, config)?),
            PretrainConfig::Bert(config) => Encoder::Bert(BertEncoder::new(vs, config)?),
            PretrainConfig::SqueezeAlbert(config) => {
                Encoder::SqueezeAlbert(SqueezeAlbertEncoder::new(vs, config)?)
            }
            PretrainConfig::SqueezeBert(config) => {
                Encoder::SqueezeBert(SqueezeBertEncoder::new(vs, config)?)
            }
            PretrainConfig::XlmRoberta(config) => Encoder::Bert(BertEncoder::new(vs, config)?),
        };

        Ok(encoder)
    }

    pub fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Vec<LayerOutput>, TransformerError> {
        match self {
            Encoder::Bert(encoder) => encoder.encode(input, attention_mask, train),
            Encoder::Albert(encoder) => encoder.encode(input, attention_mask, train),
            Encoder::SqueezeBert(encoder) => encoder.encode(input, attention_mask, train),
            Encoder::SqueezeAlbert(encoder) => encoder.encode(input, attention_mask, train),
        }
    }

    pub fn n_layers(&self) -> i64 {
        match self {
            Encoder::Bert(encoder) => encoder.n_layers(),
            Encoder::Albert(encoder) => encoder.n_layers(),
            Encoder::SqueezeBert(encoder) => encoder.n_layers(),
            Encoder::SqueezeAlbert(encoder) => encoder.n_layers(),
        }
    }
}

pub struct BertLoss {
    pub biaffine: Option<BiaffineLoss>,
    pub seq_classifiers: SequenceClassifiersLoss,
}

/// Multi-task classifier using the BERT architecture with scalar weighting.
#[derive(Debug)]
pub struct BertModel {
    biaffine: Option<BiaffineDependencyLayer>,
    embeddings: BertEmbeddingLayer,
    encoder: Encoder,
    seq_classifiers: SequenceClassifiers,
    layers_dropout: Dropout,
}

impl BertModel {
    /// Construct a fresh model.
    ///
    /// `layer_dropout` is the probability with which layers should
    /// be dropped out in scalar weighting during training.
    pub fn new<'a>(
        vs: impl Borrow<PathExt<'a>>,
        pretrain_config: &PretrainConfig,
        biaffine_config: Option<&BiaffineParserConfig>,
        n_relations: usize,
        encoders: &Encoders,
        layers_dropout: f64,
        position_embeddings: PositionEmbeddings,
    ) -> Result<Self, SyntaxDotError> {
        let vs = vs.borrow();

        let embeddings = BertEmbeddingLayer::new(vs, pretrain_config, position_embeddings)?;

        let encoder = Encoder::new(vs, pretrain_config)?;

        let biaffine = biaffine_config
            .map(|config| {
                BiaffineDependencyLayer::new(vs, pretrain_config, config, n_relations as i64)
            })
            .transpose()?;

        let seq_classifiers =
            SequenceClassifiers::new(vs, pretrain_config, encoder.n_layers(), encoders)?;

        Ok(BertModel {
            embeddings,
            encoder,
            layers_dropout: Dropout::new(layers_dropout),
            biaffine,
            seq_classifiers,
        })
    }

    /// Compute the biaffine logits for a batch of inputs from the transformer's encoding.
    pub fn biaffine_logits_from_encoding(
        &self,
        layer_outputs: &[LayerOutput],
        token_mask: &Tensor,
        train: bool,
    ) -> Result<Option<BiaffineScoreLogits>, SyntaxDotError> {
        self.biaffine
            .as_ref()
            .map(|biaffine| biaffine.forward(layer_outputs, token_mask, train))
            .transpose()
    }

    /// Encode an input.
    pub fn encode(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        train: bool,
        freeze_layers: FreezeLayers,
    ) -> Result<Vec<LayerOutput>, SyntaxDotError> {
        let start = Instant::now();

        let embeds = if freeze_layers.embeddings {
            tch::no_grad(|| self.embeddings.forward_t(inputs, train))?
        } else {
            self.embeddings.forward_t(inputs, train)?
        };

        let mut encoded = if freeze_layers.encoder {
            tch::no_grad(|| self.encoder.encode(&embeds, Some(&attention_mask), train))?
        } else {
            self.encoder.encode(&embeds, Some(&attention_mask), train)?
        };

        for layer in &mut encoded {
            *layer.output_mut() = if freeze_layers.classifiers {
                tch::no_grad(|| self.layers_dropout.forward_t(&layer.output(), train))?
            } else {
                self.layers_dropout.forward_t(&layer.output(), train)?
            };
        }

        let shape = inputs.size();
        log::debug!(
            "Encoded {} inputs with length {} in {}ms",
            shape[0],
            shape[1],
            start.elapsed().as_millis()
        );

        Ok(encoded)
    }

    /// Compute the logits for a batch of inputs.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    /// * `train`: indicates whether this forward pass will be used
    ///   for backpropagation.
    /// * `freeze_embeddings`: exclude embeddings from backpropagation.
    /// * `freeze_encoder`: exclude the encoder from backpropagation.
    pub fn logits(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        train: bool,
        freeze_layers: FreezeLayers,
    ) -> Result<HashMap<String, Tensor>, SyntaxDotError> {
        let encoding = self.encode(inputs, attention_mask, train, freeze_layers)?;
        self.seq_classifiers.forward_t(&encoding, train)
    }

    /// Compute the logits for a batch of inputs from the transformer's encoding.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    /// * `train`: indicates whether this forward pass will be used
    ///   for backpropagation.
    /// * `freeze_embeddings`: exclude embeddings from backpropagation.
    /// * `freeze_encoder`: exclude the encoder from backpropagation.
    pub fn encoder_logits_from_encoding(
        &self,
        layer_outputs: &[LayerOutput],
        train: bool,
    ) -> Result<HashMap<String, Tensor>, SyntaxDotError> {
        self.seq_classifiers.forward_t(layer_outputs, train)
    }

    /// Compute the loss given a batch of inputs and target labels.
    ///
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    /// * `token_mask`: specifies which sequence elements should be
    ///    masked when computing the loss. Typically, this is used
    ///    to exclude padding and continuation word pieces.
    /// * `targets`: the labels to be predicted, per encoder name.
    /// * `label_smoothing`: apply label smoothing, redistributing
    ///   the given probability to non-target labels.
    /// * `train`: indicates whether this forward pass will be used
    ///   for backpropagation.
    /// * `freeze_embeddings`: exclude embeddings from backpropagation.
    /// * `freeze_encoder`: exclude the encoder from backpropagation.
    #[allow(clippy::too_many_arguments)]
    pub fn loss(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        token_mask: &Tensor,
        biaffine_tensors: Option<BiaffineTensors<Tensor>>,
        targets: &HashMap<String, Tensor>,
        label_smoothing: Option<f64>,
        train: bool,
        freeze_layers: FreezeLayers,
        include_continuations: bool,
    ) -> Result<BertLoss, SyntaxDotError> {
        let encoding = self.encode(inputs, attention_mask, train, freeze_layers)?;

        if freeze_layers.classifiers {
            tch::no_grad(|| {
                let biaffine_loss = self
                    .biaffine
                    .as_ref()
                    .map(|biaffine| {
                        biaffine.loss(
                            &encoding,
                            token_mask,
                            biaffine_tensors.as_ref().unwrap(),
                            label_smoothing,
                            train,
                        )
                    })
                    .transpose()?;

                let seq_classifiers_loss = self.seq_classifiers.loss(
                    &encoding,
                    attention_mask,
                    token_mask,
                    targets,
                    label_smoothing,
                    train,
                    include_continuations,
                )?;

                Ok(BertLoss {
                    biaffine: biaffine_loss,
                    seq_classifiers: seq_classifiers_loss,
                })
            })
        } else {
            let biaffine_loss = self
                .biaffine
                .as_ref()
                .map(|biaffine| {
                    biaffine.loss(
                        &encoding,
                        token_mask,
                        biaffine_tensors.as_ref().unwrap(),
                        label_smoothing,
                        train,
                    )
                })
                .transpose()?;

            let seq_classifiers_loss = self.seq_classifiers.loss(
                &encoding,
                attention_mask,
                token_mask,
                targets,
                label_smoothing,
                train,
                include_continuations,
            )?;

            Ok(BertLoss {
                biaffine: biaffine_loss,
                seq_classifiers: seq_classifiers_loss,
            })
        }
    }

    /// Compute the top-k labels for each encoder for the input.
    ///
    /// * `token_mask`: specifies which sequence elements represent
    ///    tokens.
    /// * `attention_mask`: specifies which sequence elements should
    ///    be masked when applying the encoder.
    pub fn predict(
        &self,
        inputs: &Tensor,
        attention_mask: &Tensor,
        token_mask: &Tensor,
    ) -> Result<Predictions, SyntaxDotError> {
        let encoding = self.encode(
            inputs,
            attention_mask,
            false,
            FreezeLayers {
                embeddings: true,
                encoder: true,
                classifiers: true,
            },
        )?;

        let biaffine_score_logits = self
            .biaffine
            .as_ref()
            .map(|biaffine| biaffine.forward(&encoding, token_mask, false))
            .transpose()?;
        let sequences_top_k = self.seq_classifiers.top_k(&encoding, 3)?;

        Ok(Predictions {
            biaffine_score_logits,
            sequences_top_k,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FreezeLayers {
    pub embeddings: bool,
    pub encoder: bool,
    pub classifiers: bool,
}

#[derive(Debug)]
pub struct Predictions {
    pub biaffine_score_logits: Option<BiaffineScoreLogits>,
    pub sequences_top_k: HashMap<String, TopK>,
}
