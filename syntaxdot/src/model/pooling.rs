use serde::{Deserialize, Serialize};
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

    /// Sum and L2-normalize
    L2Sum,

    Max,

    /// Use the mean of the piece embeddings as token representation.
    Mean,
}

impl PiecePooler {
    /// Pool pieces in all layers.
    pub fn pool(
        &self,
        token_spans: &TokenSpans,
        layer_outputs: &[LayerOutput],
    ) -> Result<Tensor, TransformerError> {
        // Each output has the shape: [batch_size, 1, seq_len, hidden_size]
        let outputs = layer_outputs
            .iter()
            .map(|l| l.output().f_unsqueeze(0))
            .collect::<Result<Vec<_>, _>>()?;

        // Concatenate the layers: [layers, batch_size, seq_len, hidden_size]
        let layers = Tensor::f_cat(&outputs, 0)?;

        // Apply piece pooling: [layers, batch_size, tokens_len, hidden_size]
        self.pool_layers(&token_spans, &layers)
    }

    fn pool_layer(
        &self,
        token_spans: &TokenSpansWithRoot,
        layer: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        let (_, _layers, _, hidden_size) = layer.size4()?;

        let pooled_layer = match self {
            PiecePooler::Discard => layer.f_gather(
                1,
                &token_spans
                    .offsets()
                    .to_kind(Kind::Int64)
                    .f_unsqueeze(-1)?
                    .f_expand(&[-1, -1, hidden_size], false)?
                    // -1 is used for padding, convert into valid index.
                    .f_abs()?,
                false,
            )?,
            PiecePooler::Max => {
                let (token_embeddings, token_embeddings_mask) =
                    token_spans.embeddings_per_token(layer)?;

                let inf_mask = Tensor::from(1.0)
                    .f_sub(&token_embeddings_mask.f_to_kind(Kind::Float)?)?
                    .f_mul(&Tensor::from(-1e6))?;
                token_embeddings
                    .f_add(&inf_mask.f_unsqueeze(-1)?)?
                    .f_amax(&[2], false)?
                    .f_mul(&token_spans.token_mask()?.f_unsqueeze(-1)?)?
            }
            PiecePooler::L2Sum => {
                let (token_embeddings, _) = token_spans.embeddings_per_token(layer)?;
                let summed = token_embeddings.f_sum1(&[-2], false, Kind::Float)?;
                let norms = summed.f_norm2(2, &[-1], false)?.clamp_min(1e-9);
                summed.f_div(&norms.unsqueeze(-1))?
            }
            PiecePooler::Mean => {
                let (token_embeddings, token_embeddings_mask) =
                    token_spans.embeddings_per_token(layer)?;
                let pieces_per_token = token_embeddings_mask
                    .f_sum1(&[-1], false, Kind::Float)?
                    .f_clamp_min(1)?;
                token_embeddings
                    .f_sum1(&[3], false, Kind::Float)?
                    .f_div(&pieces_per_token.f_unsqueeze(1)?.f_unsqueeze(-1)?)?
            }
        };

        Ok(pooled_layer)
    }

    fn pool_layers(
        &self,
        token_spans: &TokenSpans,
        layers: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        let (token_embeddings, token_embeddings_mask) =
            token_spans.with_root()?.embeddings_per_token(&layers)?;

        match self {
            PiecePooler::Mean => Self::pool_mean(&token_embeddings, &token_embeddings_mask),
            PiecePooler::Discard => Self::pool_discard(&token_embeddings, &token_embeddings_mask),
            _ => unimplemented!(),
        }
    }

    fn pool_discard(
        token_embeddings: &Tensor,
        _token_embeddings_mask: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        // Slice to get the representation of the first token piece.
        Ok(token_embeddings.f_slice(3, 0, 1, 1)?.f_squeeze1(3)?)
    }

    fn pool_mean(
        token_embeddings: &Tensor,
        token_embeddings_mask: &Tensor,
    ) -> Result<Tensor, TransformerError> {
        let pieces_per_token = token_embeddings_mask
            .f_sum1(&[-1], false, Kind::Float)?
            .f_clamp_min(1)?;
        Ok(token_embeddings
            .f_sum1(&[3], false, Kind::Float)?
            .f_div(&pieces_per_token.f_unsqueeze(0)?.f_unsqueeze(-1)?)?)
    }
}

trait EmbeddingsPerToken {
    fn embeddings_per_token(
        &self,
        embeddings: &Tensor,
    ) -> Result<(Tensor, Tensor), TransformerError>;
}

impl EmbeddingsPerToken for TokenSpansWithRoot {
    fn embeddings_per_token(
        &self,
        embeddings: &Tensor,
    ) -> Result<(Tensor, Tensor), TransformerError> {
        let (layers, batch_size, _pieces_len, embed_size) = embeddings.size4()?;
        let (_batch_size, tokens_len) = self.offsets().size2()?;

        let max_token_len = i64::from(self.lens().max());

        let piece_range = Tensor::f_arange(max_token_len, (Kind::Int64, self.lens().device()))?
            .f_view([1, 1, max_token_len])?;

        let mask = piece_range.less1(&self.lens().f_unsqueeze(-1)?);

        let piece_indices = (piece_range + self.offsets().unsqueeze(-1)).f_mul(&mask)?;

        let piece_embeddings = embeddings
            .f_gather(
                2,
                &piece_indices
                    .f_view([1, batch_size, -1, 1])?
                    .f_expand(&[layers, -1, -1, embed_size], true)?,
                false,
            )?
            .f_view([layers, batch_size, tokens_len, max_token_len, embed_size])?
            .f_mul(&mask.f_view([1, batch_size, tokens_len, max_token_len, 1])?)?;

        Ok((piece_embeddings, mask))
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
            Tensor::of_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::of_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden = Tensor::arange2(36, 0, -1, (Kind::Int64, Device::Cpu))
            .view([1, 2, 9, 2])
            .to_kind(Kind::Float);

        let pooler = PiecePooler::Discard;

        let token_embeddings = pooler.pool_layers(&spans, &hidden).unwrap();

        token_embeddings.print();

        assert_eq!(
            token_embeddings,
            Tensor::of_slice2(&[
                &[36, 35, 34, 33, 30, 29, 28, 27, 0, 0, 0, 0],
                &[18, 17, 16, 15, 12, 11, 10, 9, 6, 5, 4, 3]
            ])
            .view([1, 2, 6, 2])
        );
    }

    #[test]
    fn embeddings_are_returned_per_token() {
        let spans = TokenSpansWithRoot::new(
            Tensor::of_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::of_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden = Tensor::arange2(32, 0, -1, (Kind::Int64, Device::Cpu)).view([1, 2, 8, 2]);

        let (token_embeddings, _) = spans.embeddings_per_token(&hidden).unwrap();

        assert_eq!(
            token_embeddings,
            Tensor::of_slice(&[
                30, 29, 28, 27, 26, 25, 0, 0, 24, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 13, 12, 11,
                10, 9, 0, 0, 8, 7, 6, 5, 4, 3, 0, 0, 2, 1, 0, 0
            ])
            .view([1, 2, 5, 2, 2])
        );
    }

    #[test]
    fn mean_pooler_works_correctly() {
        let spans = TokenSpans::new(
            Tensor::of_slice2(&[[1, 3, 4, -1, -1], [1, 3, 4, 6, 7]]),
            Tensor::of_slice2(&[[2, 1, 1, -1, -1], [2, 1, 2, 1, 1]]),
        );

        let hidden = Tensor::arange2(36, 0, -1, (Kind::Int64, Device::Cpu))
            .view([1, 2, 9, 2])
            .to_kind(Kind::Float);

        let pooler = PiecePooler::Mean;

        let token_embeddings = pooler.pool_layers(&spans, &hidden).unwrap();

        assert_eq!(
            token_embeddings,
            Tensor::of_slice2(&[
                &[36, 35, 33, 32, 30, 29, 28, 27, 0, 0, 0, 0,],
                &[18, 17, 15, 14, 12, 11, 9, 8, 6, 5, 4, 3]
            ])
            .view([1, 2, 6, 2])
        );
    }
}
