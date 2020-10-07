use std::ops::Deref;

use tch::Tensor;

pub enum CowTensor<'a> {
    Owned(Tensor),
    Borrowed(&'a Tensor),
}

impl<'a> Deref for CowTensor<'a> {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        match self {
            CowTensor::Owned(ref tensor) => tensor,
            CowTensor::Borrowed(tensor) => tensor,
        }
    }
}
