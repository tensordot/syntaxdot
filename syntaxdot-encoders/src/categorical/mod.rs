//! Categorical variable encoder

mod encoder;
pub use encoder::{CategoricalEncoder, ImmutableCategoricalEncoder, MutableCategoricalEncoder};

mod number;
pub use number::{ImmutableNumberer, MutableNumberer, Number};
