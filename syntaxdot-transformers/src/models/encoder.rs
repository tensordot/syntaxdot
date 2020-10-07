use tch::Tensor;

use crate::models::bert::BertLayerOutput;

/// Encoder networks.
pub trait Encoder {
    /// Apply the encoder.
    ///
    /// Returns the output and attention per layer. The (optional)
    /// attention mask of shape `[batch_size, time_steps]` indicates
    /// which tokens should be included (`true`) and excluded (`false`) from
    /// attention. This can be used to mask inactive timesteps.
    fn encode(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Vec<BertLayerOutput>;

    /// Get the number of layers that is returned by the encoder.
    fn n_layers(&self) -> i64;
}
