import torch


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, dim_input, dim_inner, dropout, pre_lnorm) -> None:
        super().__init__()

        self.dim_input = dim_input
        self.dim_inner = dim_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_inner, dim_input),
            torch.nn.Dropout(dropout),
        )

        self.layer_norm = torch.nn.LayerNorm(dim_input)

    def forward(self, x):
        if self.pre_lnorm:
            # first normalize the input, then propagate
            out = self.net(self.layer_norm(x))

            # residual connection
            out = out + x

        else:
            # first, forward the input
            out = self.net(x)

            # residual connection
            out = x + self.layer_norm(out)

        return out


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.dim = dim

    def forward(self, sequence_len, clamp=None):
        pos = torch.arange(sequence_len, dtype=torch.float).reshape(1, -1, 1)

        if clamp is not None:
            pos.clamp_(max=clamp)

        dim_ = torch.arange(self.dim, dtype=torch.float).reshape(1, 1, -1)
        phase = pos / (1e4 ** (dim_ // self.dim))

        return torch.where(dim_.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class AdaptiveEmbedding(torch.nn.Module):
    """Implementation of 'Adaptive Input Representations for Neural Language Modeling'"""

    def __init__(
        self, num_tokens, dim_embed, dim_proj, cutoffs, div_val=1, sample_softmax=False
    ) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.dim_embed = dim_embed
        self.dim_proj = dim_proj
        self.cutoffs = cutoffs + [num_tokens]
        self.div_val = div_val
        self.sample_softmax = sample_softmax

        # derivations
        self.embedding_scale = self.dim_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs

        # layers
        self.embedding_layers = torch.nn.ModuleList()
        self.embedding_projs = torch.nn.ParameterList()

        if self.div_val == 1:
            self.embedding_layers.append(
                torch.nn.Embedding(num_tokens, dim_embed, sparse=self.sample_softmax > 0)
            )

            if self.dim_proj != self.dim_embed:
                self.embedding_projs.append(
                    torch.nn.Parameter(torch.Tensor(self.dim_proj, self.dim_embed))
                )

        else:
            for i in range(len(self.cutoffs)):
                left_idx, right_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                dim_embed_ith = self.dim_embed // (self.div_val ** i)

                self.embedding_layers.append(
                    torch.nn.Embedding(right_idx - left_idx, dim_embed_ith)
                )
                self.embedding_projs.append(
                    torch.nn.Parameter(torch.Tensor(self.dim_proj, dim_embed_ith))
                )

    def forward(self, input_):
        """Forward method

        Args:
            input_ (torch.Tensor): input

        Returns:
            torch.Tensor: adaptive embedding of the word sequence
        """
        if self.div_val == 1:
            embed = self.embedding_layers[0](input_)
            if self.dim_proj != self.dim_embed:
                embed = F.linear(embed, self.embedding_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = input_.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.dim_proj], dtype=param.dtype, device=param.device
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.embedding_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.embedding_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*input_.size(), self.dim_proj)

        embed.mul_(self.embedding_scale)

        return embed
