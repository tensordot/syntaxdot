use crate::TransformerError;
use tch::Tensor;

/// Hidden layer output and attention.
#[derive(Debug)]
pub struct HiddenLayer {
    /// The output of the layer.
    pub output: Tensor,

    /// The layer attention scores (unnormalized).
    pub attention: Tensor,
}

/// Output of a BERT layer.
#[derive(Debug)]
pub enum LayerOutput {
    /// Embedding layer output.
    Embedding(Tensor),

    /// Encoder layer output.
    EncoderWithAttention(HiddenLayer),
}

impl LayerOutput {
    /// Get the layer attention.
    ///
    /// Return a `Some` value if the layer output is from an encoder layer,
    /// or `None` otherwise.
    pub fn attention(&self) -> Option<&Tensor> {
        match self {
            LayerOutput::Embedding(_) => None,
            LayerOutput::EncoderWithAttention(hidden) => Some(&hidden.attention),
        }
    }

    /// Get the embedding.
    ///
    /// Returns `Some` if the layer output is an embedding or `None`
    /// otherwise.
    pub fn embedding(&self) -> Option<&Tensor> {
        match self {
            LayerOutput::Embedding(embedding) => Some(embedding),
            LayerOutput::EncoderWithAttention(_) => None,
        }
    }

    /// Map the output representation of this layer.
    pub fn map_output<F>(&self, f: F) -> Result<Self, TransformerError>
    where
        F: Fn(&Tensor) -> Result<Tensor, TransformerError>,
    {
        let layer = match self {
            LayerOutput::Embedding(embedding) => LayerOutput::Embedding(f(embedding)?),
            LayerOutput::EncoderWithAttention(HiddenLayer { output, attention }) => {
                LayerOutput::EncoderWithAttention(HiddenLayer {
                    output: f(output)?,
                    attention: attention.shallow_clone(),
                })
            }
        };

        Ok(layer)
    }

    /// Get the layer output.
    pub fn output(&self) -> &Tensor {
        match self {
            LayerOutput::Embedding(embedding) => embedding,
            LayerOutput::EncoderWithAttention(hidden) => &hidden.output,
        }
    }

    /// Get the layer output mutably.
    pub fn output_mut(&mut self) -> &mut Tensor {
        match self {
            LayerOutput::Embedding(embedding) => embedding,
            LayerOutput::EncoderWithAttention(hidden) => &mut hidden.output,
        }
    }

    /// Get the output of an encoder layer.
    ///
    /// Return a `Some` value if the layer output is from an encoder layer,
    /// or `None` otherwise.
    pub fn encoder_with_attention(&self) -> Option<&HiddenLayer> {
        match self {
            LayerOutput::Embedding(_) => None,
            LayerOutput::EncoderWithAttention(hidden) => Some(hidden),
        }
    }
}
