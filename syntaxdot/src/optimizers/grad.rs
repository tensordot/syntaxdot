use tch::Tensor;

pub trait ZeroGrad {
    /// Zero out gradients.
    fn zero_grad(&mut self);
}

impl ZeroGrad for Vec<Tensor> {
    fn zero_grad(&mut self) {
        for tensor in self {
            if tensor.requires_grad() {
                tensor.zero_grad()
            }
        }
    }
}
