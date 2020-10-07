pub trait WordEmbeddingsConfig {
    fn dims(&self) -> i64;

    fn dropout(&self) -> f64;

    fn initializer_range(&self) -> f64;

    fn layer_norm_eps(&self) -> f64;

    fn vocab_size(&self) -> i64;
}
