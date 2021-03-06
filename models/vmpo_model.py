import torch
import numpy as np
import models.layers as layers

from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

SIZE = {
    "zero": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 0},
    "tiny": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 1},
    "small": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 3},
    "medium": {"dim": 64, "cmem_ratio": 4, "num_heads": 4, "depth": 6},
    "large": {"dim": 1024, "cmem_ratio": 4, "num_heads": 12, "depth": 12},
}

State = namedarraytuple("State", ["sequence", "memory", "compressed_memory", "length"])


class VMPOModel(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        action_size,
        linear_value_output=True,
        sequence_len=64,
        observation_normalization=True,
        size="medium",
    ) -> None:
        super().__init__()

        self.state_size = np.prod(observation_shape.state)  # memory state size
        self.action_size = action_size
        self.linear_value_output = linear_value_output
        self.sequence_len = sequence_len
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

    #! to be compatible with forward call, prev_action and prew_reward must be included
    def forward(self, observation, prev_action, prev_reward, state):
        lead_dim, T, B, _ = infer_leading_dims(observation.state, 1)
        aux_loss = None

        if T == 1:
            # model only takes one input for each batch idx
            # so, it is in eval mode (sampling)
            transformer_out, state = self.sample_forward(observation.state, state)
            value = torch.zeros(B)

        elif T == self.sequence_len:
            # model is in train mode
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

    def sample_forward(self, observation, state):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        observation = self.input_layer_norm(observation)

        device = observation.device
        if state is None:
            observations = torch.zeros((self.sequence_len, B, self.state_size), device=device)
            length = torch.zeros((1, B, 1), device=device, dtype=torch.int64)
            memory = torch.zeros(
                (self.depth, B, self.sequence_len, self.transformer_dim), device=device
            )
            compressed_memory = torch.zeros(
                (self.depth, B, self.cmem_length, self.transformer_dim), device=device
            )
        else:
            observations = state.sequence
            length = state.length.clamp_max(self.sequence_len - 1)
            memory = state.memory
            compressed_memory = state.compressed_memory

        # write new observations in tensor with older observations
        observation = observation.view(B, -1)
        # print(f'observations shape {observations.shape} observation {observation.shape}')
        # observations = torch.cat((observations, observation), dim=0)[1:]
        indexes = tuple(
            torch.cat((length[0, :], torch.arange(B, device=device).unsqueeze(-1)), dim=-1).t()
        )
        observations.index_put_(indexes, observation)

        transformer_output, new_memory, _ = self.transformer(
            observations.transpose(0, 1), layers.Memory(mem=memory, compressed_mem=None)
        )
        transformer_output = self.output_layer_norm(transformer_output).transpose(0, 1)
        # output = transformer_output.transpose(0, 1)[-1]
        output = transformer_output[length[0, :, 0], torch.arange(B)]
        # output = transformer_output[-1].reshape(T, B, -1)
        length = torch.fmod(length + 1, self.sequence_len)

        reset = (length == 0).int()[0, :, 0].reshape(B, 1, 1, 1).transpose(0, 1).expand_as(memory)
        # print(f'length {length[0, :, 0]}')
        # if B > 1 and length[0, 0, 0] != length[0, 1, 0]:
        #     breakpoint()
        memory = reset * new_memory.mem + (1 - reset) * memory
        # memory = new_memory.mem

        state = State(
            sequence=observations, length=length, memory=memory, compressed_memory=compressed_memory
        )
        return output, state


    def optim_forward(self, observation, state):
        observation = self.input_layer_norm(observation)
        output, _, aux_loss = self.transformer(observation.transpose(0, 1), layers.Memory(mem=state.memory, compressed_mem=None))
        output = self.output_layer_norm(output)
        return output.transpose(0, 1), aux_loss
