use std::ops::Div;
use std::rc::Rc;

use itertools::Itertools;
use tch::nn::{Init, Path, VarStore};
use tch::{TchError, Tensor};

/// Trait that provides the root of a variable store.
pub trait RootExt {
    /// Get the root of a variable store.
    ///
    /// In contrast to the regular `root` method, `root_ext` allows
    /// you to provide a function that maps a variable name to a
    /// parameter group. This is particularly useful for use cases
    /// where one wants to put parameters in separate groups, to
    /// give each group its own hyper-parameters.
    fn root_ext<F>(&self, parameter_group_fun: F) -> PathExt
    where
        F: 'static + Fn(&str) -> usize;
}

impl RootExt for VarStore {
    fn root_ext<F>(&self, parameter_group_fun: F) -> PathExt
    where
        F: 'static + Fn(&str) -> usize,
    {
        PathExt {
            inner: self.root(),
            parameter_group_fun: Rc::new(parameter_group_fun),
        }
    }
}

pub struct PathExt<'a> {
    inner: Path<'a>,
    parameter_group_fun: Rc<dyn Fn(&str) -> usize>,
}

impl<'a> PathExt<'a> {
    /// Create a tensor variable initialized with ones.
    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.ones(name, dims)
    }

    /// Get a sub-path of the current path.
    pub fn sub<T: ToString>(&'a self, s: T) -> PathExt<'a> {
        PathExt {
            inner: self.inner.sub(s),
            parameter_group_fun: self.parameter_group_fun.clone(),
        }
    }

    /// Create a tensor variable initialized with the given initializer.
    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Result<Tensor, TchError> {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.f_var(name, dims, init)
    }

    /// Create a tensor variable initialized with the values from another tensor.
    pub fn var_copy(&self, name: &str, t: &Tensor) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.var_copy(name, t)
    }

    /// Get the full name of `name` and return its group.
    fn name_group(&self, name: &str) -> usize {
        let fullname = format!("{}.{}", self.inner.components().join("."), name);
        (self.parameter_group_fun)(&fullname)
    }

    /// Create a tensor variable initialized with zeros.
    pub fn zeros(&self, name: &str, dims: &[i64]) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.zeros(name, dims)
    }
}

impl<'a, T> Div<T> for &'a mut PathExt<'a>
where
    T: std::string::ToString,
{
    type Output = PathExt<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}

impl<'a, T> Div<T> for &'a PathExt<'a>
where
    T: std::string::ToString,
{
    type Output = PathExt<'a>;

    fn div(self, rhs: T) -> Self::Output {
        self.sub(rhs.to_string())
    }
}
