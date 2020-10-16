use std::collections::HashMap;
use std::ops::Div;
use std::rc::Rc;

use itertools::Itertools;
use tch::nn::{Init, Path, VarStore};
use tch::Tensor;

pub trait ReassignGroups {
    fn reassign_groups<F>(&self, group_fun: F)
    where
        F: Fn(&str) -> usize;
}

impl ReassignGroups for VarStore {
    fn reassign_groups<F>(&self, group_fun: F)
    where
        F: Fn(&str) -> usize,
    {
        let mut variables = self.variables_.lock().unwrap();

        // Mapping from tensors to names.
        let tensor_names: HashMap<_, _> = variables
            .named_variables
            .iter()
            .map(|(name, tensor)| (tensor.data_ptr(), name.clone()))
            .collect();

        for var in &mut variables.trainable_variables {
            let tensor_name = &tensor_names[&var.tensor.data_ptr()];
            var.group = (group_fun)(tensor_name.as_str());
        }
    }
}

pub trait RootExt {
    fn root_ext<F>(&self, group_fun: F) -> PathExt
    where
        F: 'static + Fn(&str) -> usize;
}

impl RootExt for VarStore {
    fn root_ext<F>(&self, group_fun: F) -> PathExt
    where
        F: 'static + Fn(&str) -> usize,
    {
        PathExt {
            inner: self.root(),
            group_fun: Rc::new(group_fun),
        }
    }
}

pub struct PathExt<'a> {
    inner: Path<'a>,
    group_fun: Rc<dyn Fn(&str) -> usize>,
}

impl<'a> PathExt<'a> {
    pub fn ones(&self, name: &str, dims: &[i64]) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.ones(name, dims)
    }
    pub fn sub<T: ToString>(&'a self, s: T) -> PathExt<'a> {
        PathExt {
            inner: self.inner.sub(s),
            group_fun: self.group_fun.clone(),
        }
    }

    pub fn var(&self, name: &str, dims: &[i64], init: Init) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.var(name, dims, init)
    }

    pub fn var_copy(&self, name: &str, t: &Tensor) -> Tensor {
        let group = self.name_group(name);
        let path = self.inner.set_group(group);
        path.var_copy(name, t)
    }

    fn name_group(&self, name: &str) -> usize {
        let fullname = format!("{}.{}", self.inner.components().join("."), name);
        (self.group_fun)(&fullname)
    }

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
