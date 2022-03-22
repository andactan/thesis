import torch
import torch.nn.functional as F


class RelativeMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_input: int,
        dim_head: int,
        dropout: float = 0.0,
        dropout_attention: float = 0.0,
        pre_lnorm: bool = False,
    ) -> None:
        """Relative Multi-Head Attention Layer from Transformer-XL paper

        Args:
            num_heads (int): number of heads
            input_dim (int): input embedding size
            query_dim (int): query size
            key_dim (int): key size
            value_dim (int): value size
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """

        super().__init__()

        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_head = dim_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.dim_out = self.num_heads * self.dim_head

        # query-key-value net
        self.qkv = torch.nn.Linear(self.dim_input, 3 * self.dim_out, bias=False)

        # linear layer
        self.linear = torch.nn.Linear(self.dim_out, self.dim_input, bias=False)

        # positional embedding net
        self.r_net = torch.nn.Linear(self.dim_input, self.dim_out)

        # misc
        self.dropout = torch.nn.Dropout(dropout)
        self.dropout_attention = torch.nn.Dropout(dropout_attention)
        self.layer_norm = torch.nn.LayerNorm(self.dim_input)

        # # relative positioning weights
        # self.pos_k = torch.nn.Linear(input_dim, key_dim, bias=False)
        # self.bias_k = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)
        # self.bias_q = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        """Forward call

        Args:
            query (torch.Tensor): query input
            key (torch.Tensor): key input
            value (torch.Tensor): value input
            pos_embed (torch.Tensor): positional embeddings
        """

        # r_w_bias ==> bias_q
        # r_r_bias ==> bias_k

        # TODO remove unnecessary variables
        query_len, pos_len, batch_size = word_embed.size(0), pos_embed.size(0), word_embed.size(1)

        # append memories if not None
        input_ = word_embed
        if memories is not None:
            input_ = torch.cat([memories, word_embed], 0)

        if self.pre_lnorm:
            input_ = self.layer_norm(input_)

        qkv = self.qkv(input_)

        query_, key_, value_ = torch.chunk(qkv, 3, dim=-1)
        if memories is not None:
            query_ = query_[-query_len:]  # get activations only for sequence, freeze others

        # shift the positonal embeddings
        shifted_pos_embed = self._relative_shift(pos_embed)
        pos_key_ = self.r_net(shifted_pos_embed)

        q_bias_q = query_ + bias_q
        AC = q_bias_q.bmm(key_.transpose(1, 2))

        k_bias_k = key_ + bias_k
        BD = k_bias_k.bmm(pos_key_.transpose(1, 2))

        # compute attention probability
        pre_attn = AC + BD
        scale = key_.size(-1) ** 0.5
        attn_prob = F.softmax(pre_attn / scale, dim=-1)
        attn_prob = self.dropout_attention(attn_prob)

        # compute attention vector
        attn_vec = attn_prob.bmm(value_)

        # pass through a linear layer
        out = self.linear(attn_vec)
        out = self.dropout(out)

        if self.pre_lnorm:
            out = word_embed + out

        else:
            out = self.layer_norm(word_embed + out)

        return out

    def _relative_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x


if __name__ == "__main__":
    import misc

    print(misc.__dict__)
