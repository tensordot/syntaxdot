use serde::{Deserialize, Serialize};
use syntaxdot_transformers::models::LayerOutput;
use syntaxdot_transformers::TransformerError;
use tch::Tensor;

use crate::error::SyntaxDotError;
use crate::tensor::TokenOffsets;

/// Word/sentence piece pooler.
///
/// The models that are used in SyntaxDot use word or sentence pieces.
/// After piece tokenization, each token is represented by one initial
/// piece, followed by zero or more continuation pieces. However, in
/// sequence labeling and dependency parsing, each token must be
/// represented using a single vector (in each layer).
///
/// The pooler combines the one or more piece representations into
/// a single representation using pooling.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PiecePooler {
    /// Discard continuation pieces.
    Discard,
}

impl PiecePooler {
    /// Pool pieces in all layers.
    pub fn pool(
        &self,
        token_offsets: &TokenOffsets,
        layer_outputs: &[LayerOutput],
    ) -> Result<Vec<LayerOutput>, SyntaxDotError> {
        let mut new_layer_outputs = Vec::with_capacity(layer_outputs.len());
        for layer_output in layer_outputs {
            let new_layer_output = layer_output
                .map_output(|output| self.pool_layer(token_offsets, output))
                .map_err(SyntaxDotError::BertError)?;

            new_layer_outputs.push(new_layer_output);
        }

        Ok(new_layer_outputs)
    }

    fn pool_layer(
        &self,
        token_offsets: &TokenOffsets,
        layer: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        let (batch_size, _, hidden_size) = layer.size3()?;

        // We want to retain the first piece as the root representation.
        let root_index = Tensor::from(0)
            .expand(&[batch_size, 1], false)
            .f_to_device(token_offsets.device())?;

        let token_offsets_with_root = Tensor::f_cat(&[&root_index, token_offsets], 1)?;

        let pooled_layer = match self {
            PiecePooler::Discard => layer.f_gather(
                1,
                &token_offsets_with_root
                    .f_unsqueeze(-1)?
                    .f_expand(&[-1, -1, hidden_size], false)?
                    // -1 is used for padding, convert into valid index.
                    .f_abs()?,
                false,
            )?,
        };

        Ok(pooled_layer)
    }
}
