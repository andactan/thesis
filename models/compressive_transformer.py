import torch
import numpy as np


SIZE = {
    'zero': {
        'dim': 64,
        'cmem_ratio': 4,
        'num_heads': 4,
        'depth': 0
    },

    'tiny': {
        'dim': 64,
        'cmem_ratio': 4,
        'num_heads': 4,
        'depth': 1
    },

    'small': {
        'dim': 64,
        'cmem_ratio': 4,
        'num_heads': 4,
        'depth': 3
    },

    'medium': {
        'dim': 64,
        'cmem_ratio': 4,
        'num_heads': 4,
        'depth': 6
    },

    'large': {
        'dim': 1024,
        'cmem_ratio': 4,
        'num_heads': 12,
        'depth': 12
    },

}

class CompressiveTransformer(torch.nn.Module):
    def __init__(self, observation_shape, action_size, linear_value_output=True, sequence_length=64, observation_normalization=True, size='medium') -> None:
        super().__init__()

        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.linear_value_output = linear_value_output
        self.sequence_length = sequence_length
        self.observation_normalization = observation_normalization
        self.size = size

        # transformer configs
        self.size_dict = SIZE[self.size]
        self.transformer_dim = self.size_dict['dim']
        self.depth = self.size_dict['depth']
        self.cmem_ratio = self.size_dict['cmem_ratio']
        self.cmem_length = self.sequence_length // self.cmem_ratio
        self.transformer = 