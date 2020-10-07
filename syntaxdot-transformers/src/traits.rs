use tch::Tensor;

/// Trait to get the attention of a layer.
pub trait LayerAttention {
    /// Get the attention of a layer.
    fn layer_attention(&self) -> Option<&Tensor>;
}

/// Trait to get the output of a layer.
pub trait LayerOutput {
    /// Get the output of a layer.
    fn layer_output(&self) -> &Tensor;
}
