import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(torch.nn.Module):
    """Gets a precomputed instruction (environment name) embedding and passes through MLP

    Args:
        torch (_type_): _description_
    """

    def __init__(self, dim_input, dim_hidden, dim_output) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, x):
        return self.model(x)