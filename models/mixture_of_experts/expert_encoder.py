# https://github.com/facebookresearch/mtrl

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixture_of_experts.state_encoder import SelectionNetwork, StateEncoder, StateEncoders


class ExpertEncoder(nn.Module):
    def __init__(
        self,
        state_encoder_dim_output,
        state_encoder_dim_input,
        state_encoder_dim_hidden,
        weight_network_dim_input,
        weight_network_dim_hidden,
        weight_network_dim_output,
        num_experts,
        ):
        super().__init__()

        self.state_encoder = StateEncoders(
            dim_input=state_encoder_dim_input,
            dim_hidden=state_encoder_dim_hidden,
            dim_output=state_encoder_dim_output,
            num_experts=num_experts
        )

        self.weight_network = SelectionNetwork(
            dim_input=weight_network_dim_input,
            dim_hidden=weight_network_dim_hidden,
            dim_output=weight_network_dim_output
        )

    def forward(self, state, context_embedding):

        mixture_of_experts = self.state_encoder(state) # seq, expert, batch, dim

        # detach the context embedding from the computation graph
        context_embedding = context_embedding.detach()
        weights_out = self.weight_network(context_embedding)
        weights = F.softmax(weights_out, dim=-1)

        num_dims_moe = list(range(len(mixture_of_experts.shape)))
        if len(num_dims_moe) <= 3:
            mixture_of_experts = mixture_of_experts.permute(*num_dims_moe[::-1])

        elif len(num_dims_moe) == 4:
            mixture_of_experts = mixture_of_experts.permute(3, 1, 2, 0)

        sum_weights = weights.sum(dim=-1)
        sum_mixture_of_experts_after_weights = (mixture_of_experts * weights).sum(dim=-1)
        encoding = sum_mixture_of_experts_after_weights / sum_weights # unnecessary (division by 1?)

        num_dims_encoding = list(range(len(encoding.shape)))
        out = encoding.permute(*num_dims_encoding[1:], num_dims_encoding[0])
        
        return out

# e = ExpertEncoder(
#     state_encoder_dim_input=50, 
#     state_encoder_dim_hidden=50, 
#     state_encoder_dim_output=50,
#     weight_network_dim_input=100,
#     weight_network_dim_hidden=50,
#     weight_network_dim_output=2,
#     num_experts=2)

<<<<<<< Updated upstream
# state = torch.rand(128, 50)
# context = torch.rand(128, 100)

# # torch.cat((state, context), dim=-1)
=======
        attention = F.softmax(attention, dim=2)
        combined = torch.matmul(attention.permute(0, 1, 3, 2), temp)
        combined = combined.squeeze(dim=2)
        normalizer = combined.sum(dim=-1)
        combined = combined / normalizer
        return self.mlp(combined)

e = ExpertEncoder(state_encoder_dim_input=39, state_encoder_dim_hidden=50, state_encoder_dim_output=100, num_experts=2)
state = torch.rand(14, 14, 39)
context = torch.rand(14, 14, 100)
>>>>>>> Stashed changes

x = e(state, context)
print()
# z = state.matmul(weight)
# y = state.squeeze(dim=0).matmul(weight)
# print('z', z)
# print('y', y)
# attention = e(state, context)
# # print(attention.shape)

# # if len(state.shape) == 2:
# #     attention = attention.squeeze(dim=0)
# #     z = torch.cat((attention, context), dim=1)

# # if len(state.shape) == 3:
# #     z = torch.cat((attention, context), dim=2)
# # print(z.shape)

# # # state = torch.rand(1, 50)
# # # state = state.view(1, -1)

# # # print(state.shape)








