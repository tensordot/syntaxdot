use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use syntaxdot_tch_ext::tensor::SumDim;
use syntaxdot_transformers::models::LayerOutput;
use syntaxdot_transformers::TransformerError;
use tch::{Kind, Tensor};

use crate::error::SyntaxDotError;
use crate::tensor::{TokenSpans, TokenSpansWithRoot};

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

    /// Use the mean of the piece representations as token representations.
    Mean,
}

impl PiecePooler {
    /// Pool pieces in all layers.
    pub fn pool(
        &self,
        token_spans: &TokenSpans,
        layer_outputs: &[LayerOutput],
    ) -> Result<Vec<LayerOutput>, SyntaxDotError> {
        let mut new_layer_outputs = Vec::with_capacity(layer_outputs.len());

        // Note: it would seem more efficient to stack all layers and pool all layers
        // at the same time. However, this leads to significant memory overhead, given
        // the shape [layers, batch_size, tokens_len, max_token_len, hidden], having
        // one token consisting of many pieces blows up the memory use. We control the
        // factor `layers` factor by pooling layer-wise.
        //
        // Todo: we could precompute the piece indices once and then only gather once
        // per layer.
        for layer_output in layer_outputs {
            let new_layer_output = layer_output
                .map_output(|output| self.pool_layer(&token_spans.with_root()?, output))
                .map_err(SyntaxDotError::BertError)?;

            new_layer_outputs.push(new_layer_output);
        }

        Ok(new_layer_outputs)
    }

    /// Pool the pieces in a single layer.
    fn pool_layer(
        &self,
        token_spans: &TokenSpansWithRoot,
        layer: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        let token_embeddings = token_spans.embeddings_per_token(layer)?;

        let pooled_layer = match self {
            PiecePooler::Discard => Self::pool_discard(&token_embeddings)?,
            PiecePooler::Mean => Self::pool_mean(&token_embeddings)?,
        };

        Ok(pooled_layer)
    }

    /// Discard pooling.
    fn pool_discard(token_embeddings: &TokenEmbeddings) -> Result<Tensor, TransformerError> {
        Ok(token_embeddings
            .embeddings
            .f_slice(2, 0, 1, 1)?
            .f_squeeze_dim(2)?)
    }

    /// Mean pooling
    fn pool_mean(token_embeddings: &TokenEmbeddings) -> Result<Tensor, TransformerError> {
        let pieces_per_token = token_embeddings
            .mask
            .f_sum_dim(2, false, Kind::Float)?
            .f_clamp_min(1)?;
        Ok(token_embeddings
            .embeddings
            .f_sum_dim(2, false, Kind::Float)?
            .f_div(&pieces_per_token.f_unsqueeze(-1)?)?)
    }
}

#[derive(Debug)]
struct TokenEmbeddings {
    embeddings: Tensor,
    mask: Tensor,
}

/// Piece embeddings per token.
trait EmbeddingsPerToken {
    /// Get the piece embeddings for each token.
    ///
    /// This method takes a batch of piece embeddings and partitions the
    /// embeddings per token.
    ///
    /// The input of this method is a batch of pieces with the shape
    /// `[batch_size, pieces_len, hidden_size]` and partitions the pieces
    /// by token with shape `[batch_size, tokens_len, max_token_len, hidden_size]`,
    /// where `max_token_len` is the maximum length of a token in pieces. For
    /// tokens that are shorter than the maximum length, the remaining padding
    /// can be discarded with the mask of shape
    /// `[batch_size, tokens_len, max_token_len]`.
    fn embeddings_per_token(
        &self,
        embeddings: &Tensor,
    ) -> Result<TokenEmbeddings, TransformerError>;
}

impl EmbeddingsPerToken for TokenSpansWithRoot {
    fn embeddings_per_token(
        &self,
        embeddings: &Tensor,
    ) -> Result<TokenEmbeddings, TransformerError> {
        let (batch_size, _pieces_len, embed_size) = embeddings.size3()?;
        let (_batch_size, tokens_len) = self.offsets().size2()?;

        let max_token_len = i64::try_from(&self.lens().max())?;

        let piece_range = Tensor::f_arange(max_token_len, (Kind::Int64, self.lens().device()))?
            .f_view([1, 1, max_token_len])?;

        let mask = piece_range.less_tensor(&self.lens().f_unsqueeze(-1)?);

        let piece_indices = (piece_range + self.offsets().unsqueeze(-1)).f_mul(&mask)?;

        let piece_embeddings = embeddings
            .f_gather(
                1,
                &piece_indices
                    .f_view([batch_size, -1, 1])?
                    .f_expand(&[-1, -1, embed_size], true)?,
                false,
            )?
            .f_view([batch_size, tokens_len, max_token_len, embed_size])?
            .f_mul(&mask.f_unsqueeze(-1)?)?;

        Ok(TokenEmbeddings {
            embeddings: piece_embeddings,
            mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use crate::model::pooling::{EmbeddingsPerToken, PiecePooler};
    use crate::tensor::{TokenSpans, TokenSpansWithRoot};

    #[test]
    fn discard_pooler_works_correctly() {
        let spans = TokenSpans::new(
            Tensor::from_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::from_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden = Tensor::arange_start_step(36, 0, -1, (Kind::Int64, Device::Cpu))
            .view([2, 9, 2])
            .to_kind(Kind::Float);

        let pooler = PiecePooler::Discard;

        let token_embeddings = pooler
            .pool_layer(&spans.with_root().unwrap(), &hidden)
            .unwrap();

        assert_eq!(
            token_embeddings,
            Tensor::from_slice2(&[
                &[36, 35, 34, 33, 30, 29, 28, 27, 0, 0, 0, 0],
                &[18, 17, 16, 15, 12, 11, 10, 9, 6, 5, 4, 3]
            ])
            .view([2, 6, 2])
        );
    }

    #[test]
    fn embeddings_are_returned_per_token() {
        let spans = TokenSpansWithRoot::new(
            Tensor::from_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::from_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden =
            Tensor::arange_start_step(32, 0, -1, (Kind::Int64, Device::Cpu)).view([2, 8, 2]);

        let token_embeddings = spans.embeddings_per_token(&hidden).unwrap();

        assert_eq!(
            token_embeddings.embeddings,
            Tensor::from_slice(&[
                30, 29, 28, 27, 26, 25, 0, 0, 24, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 13, 12, 11,
                10, 9, 0, 0, 8, 7, 6, 5, 4, 3, 0, 0, 2, 1, 0, 0
            ])
            .view([2, 5, 2, 2])
        );

        assert_eq!(
            token_embeddings.mask,
            Tensor::from_slice(&[
                true, true, true, false, true, false, false, false, false, false, true, true, true,
                false, true, true, true, false, true, false
            ])
            .view([2, 5, 2])
        );
    }

    #[test]
    fn mean_pooler_works_correctly() {
        let spans = TokenSpans::new(
            Tensor::from_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::from_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden = Tensor::arange_start_step(36, 0, -1, (Kind::Int64, Device::Cpu))
            .view([2, 9, 2])
            .to_kind(Kind::Float);

        let pooler = PiecePooler::Mean;

        let token_embeddings = pooler
            .pool_layer(&spans.with_root().unwrap(), &hidden)
            .unwrap();

        assert_eq!(
            token_embeddings,
            Tensor::from_slice2(&[
                &[36, 35, 33, 32, 30, 29, 28, 27, 0, 0, 0, 0,],
                &[18, 17, 15, 14, 12, 11, 9, 8, 6, 5, 4, 3]
            ])
            .view([2, 6, 2])
        );
    }
}
