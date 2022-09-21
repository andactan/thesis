import torch
import torch.nn as nn
import torch.nn.functional as F

from state_encoder import StateEncoder


class ExpertEncoder(nn.Module):
    def __init__(
        self,
        state_encoder_dim_output,
        state_encoder_dim_input,
        state_encoder_dim_hidden,
        num_experts,
        # dim_input_context_encoder,
        # dim_hidden_context_encoder,
        # dim_output_context_encoder,
        ):
        super().__init__()

        # self.context_encoder = ContextEncoder(
        #     dim_input=dim_input_context_encoder,
        #     dim_hidden=dim_hidden_context_encoder,
        #     dim_output=dim_output_context_encoder
        # )

        self.state_encoder = StateEncoder(
            dim_input=state_encoder_dim_input,
            dim_hidden=state_encoder_dim_hidden,
            dim_output=state_encoder_dim_output,
            num_experts=num_experts
        )

        self.mlp = nn.Sequential(
            nn.Linear(state_encoder_dim_output, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 25)
        )

    def forward(self, state, context_embedding):

        # add sequence_len as leading dimension
        if len(state.shape) == 2:
            state = state.unsqueeze(dim=0)
            context_embedding = context_embedding.unsqueeze(dim=0)

        mixture_of_experts = self.state_encoder(state) # n x num_experts
        # context = self.context_encoder(embedding) # n x dim_output

        # detach the context embedding from the computation graph
        context_embedding = context_embedding.detach() # n x dim_output

        temp = mixture_of_experts.permute(1, 0, 2, 3) # expert, seq, batch, dim
        ctx_emb = context_embedding.unsqueeze(dim=1).permute(0, 2, 1, 3)
        attention = torch.matmul(temp, ctx_emb)

        attention = F.softmax(attention, dim=1)
        combined = torch.matmul(temp.permute(0, 2, 1), attention)
        combined = combined.squeeze(dim=2)
        return self.mlp(combined)

e = ExpertEncoder(state_encoder_dim_input=50, state_encoder_dim_hidden=50, state_encoder_dim_output=20, num_experts=2)
state = torch.rand(4, 3, 50)
context = torch.rand(4, 3, 20)

attention = e(state, context)
print(attention.shape)

z = torch.cat((attention, context), dim=1)
print(z.shape)

# state = torch.rand(1, 50)
# state = state.view(1, -1)

# print(state.shape)








