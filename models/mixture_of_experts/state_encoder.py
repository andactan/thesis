
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class CustomLinear(nn.Module):
    """Helper module to aggregate experts in a single container
    """

    def __init__(self, num_experts: int, dim_input: int, dim_output: int):
        super().__init__()

        self.num_experts = num_experts
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.weight = nn.Parameter(
            torch.rand(self.num_experts, self.dim_input, self.dim_output)
        )
        self.bias = nn.Parameter(
            torch.rand(self.num_experts, 1, self.dim_output)
        )

    def forward(self, x):
        return x.matmul(self.weight) + self.bias

# c = nn.Parameter(torch.rand(4, 50, 25))
# print(c.shape)
# b = nn.Parameter(torch.rand(4, 25, 25))
# x = torch.rand(1, 50)

# y = x.matmul(c)
# z = y.matmul(b)
# print(z.shape)



class StateEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_experts):
        super().__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_experts = num_experts

        self.encoder = nn.Sequential(OrderedDict([
            ('input_layer', CustomLinear(self.num_experts, self.dim_input, self.dim_hidden)),
            ('relu1', nn.ReLU()),
            ('hidden_1', CustomLinear(self.num_experts, self.dim_hidden, self.dim_hidden)),
            ('relu2', nn.ReLU()),
            ('output', CustomLinear(self.num_experts, self.dim_hidden, self.dim_output))
        ]))

    def forward(self, x):
        if len(x.shape) == 2:
            out = x.unsqueeze(dim=0)

        elif len(x.shape) == 3:
            out = x.unsqueeze(dim=1)

        else:
            out = x

        return self.encoder(out)

# e = StateEncoder(dim_input=50, dim_hidden=50, dim_output=25, num_experts=4)
# x = torch.rand(2, 1, 50) # sequence_len, batch, dim
# y = e(x)
# print(y.shape)


