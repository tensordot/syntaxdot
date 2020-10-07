use tch::nn::VarStore;

pub trait ZeroGrad {
    /// Zero out gradients.
    fn zero_grad(&self);
}

impl ZeroGrad for VarStore {
    fn zero_grad(&self) {
        for (_, mut tensor) in self.variables() {
            if tensor.requires_grad() {
                tensor.zero_grad()
            }
        }
    }
}
