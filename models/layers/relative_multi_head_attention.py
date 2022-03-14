import torch
import torch.nn.functional as F
from models.layers.multi_head_attention import AttentionHead, MultiHeadAttentionLayer
from models.layers.misc import FeedForward


class RelativeMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_input: int,
        dim_head: int,
        dropout: float = 0.1,
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

        query_len, pos_len, batch_size = word_embed.size(0), pos_embed.size(0), word_embed.size(1)

        # append memories if not None
        input_ = word_embed
        if memories is not None:
            input_ = torch.cat([memories, word_embed], 0)

        if self.pre_lnorm:
            input_ = self.pre_lnorm(input_)

        qkv = self.qkv(input_)
        r = self.r_net(pos_embed)

        query, key, value = torch.chunk(qkv, 3, dim=-1)
        if memories is not None:
            query = query[-query_len:]  # get activations only for sequence, freeze others

        # shift the positonal embeddings
        shifted_pos_embed = self._relative_shift(pos_embed)
        query_ = self.q(query)
        key_ = self.k(key)
        value_ = self.v(value)
        pos_key_ = self.pos_k(shifted_pos_embed)

        q_bias_q = query_ + bias_q
        AC = q_bias_q.bmm(key_.transpose(1, 2))

        k_bias_k = key_ + bias_k
        BD = k_bias_k.bmm(pos_key_.transpose(1, 2))

        # calculate attention
        pre_attn = AC + BD
        scale = key.size(-1) ** 0.5
        attn = F.softmax(pre_attn / scale, dim=-1)

        return attn.bmm(value_)

    def _relative_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x


class RelativeAttentionDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_input: int,
        dim_head: int,
        dim_ff: int,
        dropout: float = 0.1,
        pre_lnorm: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.attention = RelativeMultiHeadAttention(
            num_heads=self.num_heads,
            dim_input=self.dim_input,
            dim_head=self.dim_head,
            dropout=self.dropout,
            pre_lnorm=self.pre_lnorm,
        )

        self.ff = FeedForward(
            input_dim=self.dim_input, output_dim=self.dim_input, hidden_dim=self.dim_ff
        )

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        out = self.attention(
            word_embed, pos_embed, bias_q, bias_k, attention_mask=attention_mask, memories=memories
        )

        out = self.ff(out)

        return out
