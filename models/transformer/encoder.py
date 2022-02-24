import torch
import torch.nn.functional as F

from models.transformer.misc import Residual
from models.transformer.multi_head_attention import MultiHeadAttentionLayer

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim=512, num_heads=6, query_dim=None, key_dim=None, value_dim=None, dropout=0.1, feedforward_dim=2048) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.dropout = dropout
        self.feedforward_dim = feedforward_dim

        # multihead attention configs
        default_dim = max(input_dim // num_heads, 1)
        self.query_dim = query_dim or default_dim
        self.key_dim = key_dim or default_dim
        self.value_dim = value_dim or default_dim

        # attention layer
        self.attention_layer = Residual(
            MultiHeadAttentionLayer(num_heads, input_dim, query_dim, key_dim, value_dim)
        )

