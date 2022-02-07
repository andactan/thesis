from email import policy
import torch
import numpy as np
import layers

from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


SIZE = {
    "zero": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 0},
    "tiny": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 1},
    "small": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 3},
    "medium": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 6},
    "large": {"dim": 1024, "cmem_ratio": 4, "num_heads": 12, "depth": 12},
}


class CompressiveTransformer(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        action_size,
        linear_value_output=True,
        sequence_length=64,
        observation_normalization=True,
        size="medium",
    ) -> None:
        super().__init__()

        self.state_size = np.prod(observation_shape.state)
        self.action_size = action_size
        self.linear_value_output = linear_value_output
        self.sequence_len = sequence_length
        self.observation_normalization = observation_normalization
        self.size = size

        # transformer configs
        self.size_dict = SIZE[self.size]
        self.transformer_dim = self.size_dict["dim"]
        self.depth = self.size_dict["depth"]
        self.cmem_ratio = self.size_dict["cmem_ratio"]
        self.cmem_length = self.sequence_len // self.cmem_ratio
        self.transformer = layers.CompressiveTransformer(
            num_tokens=20000,
            dim=self.transformer_dim,
            embedding_dim=self.state_size,
            heads=SIZE[self.size]["num_heads"],
            depth=self.depth,
            sequence_len=self.sequence_len,
            memory_len=self.sequence_len,
            memory_layers=range(1, self.depth + 1),
        )
        self.transformer.token_embedding = torch.nn.Identity()
        self.transformer.to_logits = torch.nn.Identity()

        # input layer normalization
        if self.observation_normalization:
            self.input_layer_norm = torch.nn.LayerNorm(self.state_size)
        else:
            self.input_layer_norm = torch.nn.Identity()

        # output layer normalization
        self.output_layer_norm = torch.nn.Identity()

        self.softplus = torch.nn.Softplus()

        # policy network
        self.policy_net = MlpModel(
            input_size=self.transformer_dim, hidden_sizes=256, output_size=2 * self.action_size
        )

        # value network
        self.value_net = MlpModel(
            input_size=self.transformer_dim,
            hidden_sizes=256,
            output_size=1 if self.linear_value_output else None,
        )
        self.mask = torch.ones((self.sequence_len, self.sequence_len), dtype=torch.int8).triu()

    def forward(self, observation, state):
        lead_dim, T, B, _ = infer_leading_dims(observation.shape, 1)
        aux_loss = None

        if T == 1:
            transformer_out, state = self.sample_forward(observation.state, state)
            value = torch.zeros(B)

        elif T == self.sequence_len:
            transformer_out, aux_loss = self.optim_forward(observation.state, state)
            value = self.value_net(transformer_out).reshape(T * B, -1)

        else:
            raise NotImplementedError

        policy_out = self.policy_net(transformer_out).view(T * B, -1)
        mu = torch.tanh(policy_out[:, self.action_size :])
        std = self.softplus(policy_out[:, : self.action_size])

        # reshape tensors
        mu, std, value = restore_leading_dims((mu, std, value), lead_dim, T, B)

        return mu, std, value, state, aux_loss
