import torch
from models.transformer_xl.relative_multi_head_attention import RelativeMultiHeadAttention
from models.transformer_xl.misc import PositionwiseFeedForward


class RelativeMultiHeadAttentionDecoderLayer(torch.nn.Module):
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

        self.ff = PositionwiseFeedForward(dim_input=self.dim_input, dim_inner=self.dim_ff)

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        out = self.attention(
            word_embed, pos_embed, bias_q, bias_k, attention_mask=attention_mask, memories=memories
        )

        out = self.ff(out)

        return out
