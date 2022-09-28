import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class StateEncoder(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output):
    super().__init__()

    self.model = nn.Sequential(OrderedDict([
      ('input', nn.Linear(dim_input, dim_hidden, bias=True)),
      ('relu', nn.ReLU()),
      ('output', nn.Linear(dim_hidden, dim_output, bias=True))
    ]))

  def forward(self, x):
    return self.model(x)


class StateEncoders(nn.Module):
  def __init__(self, num_experts, dim_hidden, dim_input, dim_output):
    super().__init__()
    self.encoders = nn.ModuleList([
      StateEncoder(dim_input=dim_input, dim_hidden=dim_hidden, dim_output=dim_output)
      for _ in range(num_experts)
    ])

  def forward(self, x):
    outs = []
    for idx, encoder in enumerate(self.encoders):
      outs.append(encoder(x).unsqueeze(dim=0))

    return torch.cat(outs, dim=0)

class SelectionNetwork(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output):
    super().__init__()
    self.model = nn.Sequential(OrderedDict([
      ('linear1', nn.Linear(dim_input, dim_hidden)),
      ('relu1', nn.ReLU()),
      ('linear2', nn.Linear(dim_hidden, dim_hidden)),
      ('relu2', nn.ReLU()),
      ('output', nn.Linear(dim_hidden, dim_output))
    ]))

  def forward(self, x):
    return self.model(x)