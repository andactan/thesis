import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.multi_head_attention import AttentionHead, MultiHeadAttentionLayer


class RelativeAttentionHead(AttentionHead):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Relative Multi-Head Attention Layer from Transformer-XL paper

        Args:
            num_heads (int): number of heads
            input_dim (int): input embedding
            query_dim (int): query size
            key_dim (int): key size
            value_dim (int): value size
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__(
            num_heads=num_heads,
            input_dim=input_dim,
            query_dim=query_dim,
            value_dim=value_dim,
            key_dim=key_dim,
        )

        self.dropout = dropout

        # relative positioning weights
        self.pos_k = torch.nn.Linear(input_dim, key_dim, bias=False)
        self.bias_k = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)
        self.bias_q = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)

    def forward(self, query, key, value, pos_embed):
        """Forward call

        Args:
            query (torch.Tensor): query input
            key (torch.Tensor): key input
            value (torch.Tensor): value input
            pos_embed (torch.Tensor): positional embeddings
        """
        
        # shift the positonal embeddings
        shifted_pos_embed = self._relative_shift(pos_embed)
        query_ = self.q(query)
        key_ = self.k(key)
        value_ = self.v(value)
        pos_key_ = self.pos_k(shifted_pos_embed)

        q_bias_q = query_ + self.bias_q
        AC = q_bias_q.bmm(key_.transpose(1, 2))
        
        k_bias_k = key_ + self.bias_k
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
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x


class RelativeMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self, num_heads, input_dim, query_dim, key_dim, value_dim) -> None:
        super().__init__(num_heads, input_dim, query_dim, key_dim, value_dim)

        self.heads = nn.ModuleList([RelativeAttentionHead()])
