//! Dependency encoding/decoding for biaffine parsing.

mod encoder;
pub use encoder::{
    DependencyEncoding, EncodeError, ImmutableDependencyEncoder, MutableDependencyEncoder,
};

#[doc(hidden)]
pub mod mst;
