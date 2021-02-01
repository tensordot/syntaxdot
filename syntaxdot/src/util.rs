use rand::Rng;
use tch::{Kind, Tensor};

use crate::error::SyntaxDotError;

pub struct RandomRemoveVec<T, R> {
    inner: Vec<T>,
    rng: R,
}

impl<T, R> RandomRemoveVec<T, R>
where
    R: Rng,
{
    /// Create a shuffler with the given capacity.
    pub fn with_capacity(capacity: usize, rng: R) -> Self {
        RandomRemoveVec {
            inner: Vec::with_capacity(capacity + 1),
            rng,
        }
    }

    /// Check whether the shuffler is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Push an element into the shuffler.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Get the number of elements in the shuffler.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, R> RandomRemoveVec<T, R>
where
    R: Rng,
{
    /// Randomly remove an element from the shuffler.
    pub fn remove_random(&mut self) -> Option<T> {
        if self.inner.is_empty() {
            None
        } else {
            Some(
                self.inner
                    .swap_remove(self.rng.gen_range(0, self.inner.len())),
            )
        }
    }

    /// Add `replacement` to the inner and randomly remove an element.
    ///
    /// `replacement` could also be drawn randomly.
    pub fn push_and_remove_random(&mut self, replacement: T) -> T {
        self.inner.push(replacement);
        self.inner
            .swap_remove(self.rng.gen_range(0, self.inner.len()))
    }
}

/// Convert sequence lengths to masks.
pub fn seq_len_to_mask(seq_lens: &Tensor, max_len: i64) -> Result<Tensor, SyntaxDotError> {
    let batch_size = seq_lens.size()[0];
    Ok(Tensor::f_arange(max_len, (Kind::Int, seq_lens.device()))?
        // Construct a matrix [batch_size, max_len] where each row
        // is 0..(max_len - 1).
        .f_repeat(&[batch_size])?
        .f_view_(&[batch_size, max_len])?
        // Time steps less than the length in seq_lens are active.
        .f_lt_1(&seq_lens.unsqueeze(1))?
        // For some reason the kind is Int?
        .to_kind(Kind::Bool))
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use super::RandomRemoveVec;

    #[test]
    fn random_remove_vec() {
        let mut rng = XorShiftRng::seed_from_u64(42);
        let mut elems = RandomRemoveVec::with_capacity(3, XorShiftRng::seed_from_u64(42));
        elems.push(1);
        elems.push(2);
        elems.push(3);

        // Before: [1 2 3]
        assert_eq!(rng.gen_range(0, 4 as usize), 1);
        assert_eq!(elems.push_and_remove_random(4), 2);

        // Before: [1 4 3]
        assert_eq!(rng.gen_range(0, 4 as usize), 2);
        assert_eq!(elems.push_and_remove_random(5), 3);

        // Before: [1 4 5]
        assert_eq!(rng.gen_range(0, 4 as usize), 1);
        assert_eq!(elems.push_and_remove_random(6), 4);

        // Before: [1 6 5]
        assert_eq!(rng.gen_range(0, 3 as usize), 1);
        assert_eq!(elems.remove_random().unwrap(), 6);

        // Before: [1 5]
        assert_eq!(rng.gen_range(0, 2 as usize), 0);
        assert_eq!(elems.remove_random().unwrap(), 1);

        // Before: [5]
        assert_eq!(rng.gen_range(0, 1 as usize), 0);
        assert_eq!(elems.remove_random().unwrap(), 5);

        // Exhausted
        assert_eq!(elems.remove_random(), None);

        // The buffer is empty, so always return the next number
        assert_eq!(elems.push_and_remove_random(7), 7);
        assert_eq!(elems.push_and_remove_random(8), 8);
    }
}
