pub mod config;

pub mod dataset;

pub mod error;

pub mod encoders;

pub mod lr;

pub mod model;

pub mod optimizers;

pub mod tagger;

pub mod tensor;

pub mod util;

/// The syntaxdot version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
