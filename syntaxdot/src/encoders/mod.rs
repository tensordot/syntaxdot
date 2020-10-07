//! Encoder configuration and construction.

mod config;
pub use config::{DependencyEncoder, EncoderType, EncodersConfig, NamedEncoderConfig};

#[allow(clippy::module_inception)]
mod encoders;
pub use encoders::{DecoderError, Encoder, EncoderError, Encoders, NamedEncoder};
