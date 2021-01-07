use std::cell::RefCell;
use std::hash::Hash;

use numberer::Numberer;
use serde_derive::{Deserialize, Serialize};

/// Number a categorical variable.
#[allow(clippy::len_without_is_empty)]
pub trait Number<V>
where
    V: Clone + Eq + Hash,
{
    /// Construct a numberer for categorical variables.
    fn new(numberer: Numberer<V>) -> Self;

    /// Get the number of possible values in the categorical variable.
    ///
    /// This includes reserved numerical representations that do
    /// not correspond to values in the categorial variable.
    fn len(&self) -> usize;

    /// Get the number of a value from a categorical variable.
    ///
    /// Mutable implementations of this trait must add the value if it
    /// is unknown and always return [`Option::Some`].
    fn number(&self, value: V) -> Option<usize>;

    /// Get the value corresponding of a number.
    ///
    /// Returns [`Option::None`] if the number is unknown *or* a
    /// reserved number.
    fn value(&self, number: usize) -> Option<V>;
}

/// An immutable categorical variable numberer.
#[derive(Deserialize, Serialize)]
pub struct ImmutableNumberer<V>(Numberer<V>)
where
    V: Clone + Eq + Hash;

impl<V> Number<V> for ImmutableNumberer<V>
where
    V: Clone + Eq + Hash,
{
    fn new(numberer: Numberer<V>) -> Self {
        ImmutableNumberer(numberer)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn number(&self, value: V) -> Option<usize> {
        self.0.number(&value)
    }

    fn value(&self, number: usize) -> Option<V> {
        self.0.value(number).cloned()
    }
}

/// A mutable categorical variable numberer using interior mutability.
#[derive(Deserialize, Serialize)]
pub struct MutableNumberer<V>(RefCell<Numberer<V>>)
where
    V: Clone + Eq + Hash;

impl<V> Number<V> for MutableNumberer<V>
where
    V: Clone + Eq + Hash,
{
    fn new(numberer: Numberer<V>) -> Self {
        MutableNumberer(RefCell::new(numberer))
    }

    fn len(&self) -> usize {
        self.0.borrow().len()
    }

    fn number(&self, value: V) -> Option<usize> {
        Some(self.0.borrow_mut().add(value))
    }

    fn value(&self, number: usize) -> Option<V> {
        self.0.borrow().value(number).cloned()
    }
}
