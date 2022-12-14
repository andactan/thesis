import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class ContextEncoder(nn.Module):
  """ Project the context embedding to a lower dimension
  """
  def __init__(self, dim_input, dim_hidden, dim_output):
    super().__init__()
    self.model = nn.Sequential(OrderedDict([
      ('input', nn.Linear(dim_input, dim_hidden)),
      ('relu1', nn.ReLU()),
      ('linear1', nn.Linear(dim_hidden, dim_hidden)),
      ('relu', nn.ReLU()),
      ('trunk_linear1', nn.Linear(dim_hidden, dim_hidden)),
      ('relu3', nn.ReLU()),
      ('trunk_linear2', nn.Linear(dim_hidden, dim_output)),
      ('relu4', nn.ReLU()),
      ('output', nn.Linear(dim_hidden, dim_output))
    ]))

  def forward(self, x):
    return self.model(x)