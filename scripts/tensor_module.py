from torch import nn

class TensorModule(nn.Module):
    def __init__(self, tensors):
        super(TensorModule, self).__init__()

        for tensor_name, tensor in tensors.items():
            setattr(self, tensor_name, nn.Parameter(tensor))

