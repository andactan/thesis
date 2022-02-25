import torch

from models.transformer.multi_head_attention import MultiHeadAttentionLayer
from models.transformer.misc import FeedForward, Residual
from models.transformer.utils import positional_encoding


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        input_dim=512,
        output_dim=512,
        num_heads=6,
        query_dim=None,
        key_dim=None,
        value_dim=None,
        dropout=0.1,
        feedforward_dim=2048,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.feedforward_dim = feedforward_dim

        # multihead attention configs
        default_dim = max(self.input_dim // self.num_heads, 1)
        self.query_dim = query_dim or default_dim
        self.key_dim = key_dim or default_dim
        self.value_dim = value_dim or default_dim

        # first attention layer
        self.attention_layer_0 = Residual(
            MultiHeadAttentionLayer(
                self.num_heads, self.input_dim, self.query_dim, self.key_dim, self.value_dim
            ),
            self.input_dim,
            self.dropout,
        )

        # second attention layer
        self.attention_layer_1 = Residual(
            MultiHeadAttentionLayer(
                self.num_heads, self.input_dim, self.query_dim, self.key_dim, self.value_dim
            ),
            self.input_dim,
            self.dropout,
        )

        # feedforward layer
        self.feed_forward_layer = Residual(
            FeedForward(
                input_dim=self.input_dim,
                hidden_dim=self.feedforward_dim,
                output_dim=self.output_dim,
            ),
            self.input_dim,
            self.dropout,
        )

    def forward(self, x, y):
        out = self.attention_layer_0(x, x, x)
        out = self.attention_layer_1(out, y, y)

        return self.feed_forward_layer(out)


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        num_layers=6,
        num_heads=8,
        input_dim=512,
        output_dim=512,
        feedforward_dim=2048,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    num_heads=self.num_heads,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    feedforward_dim=self.feedforward_dim,
                    dropout=self.dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, y):
        sequence_len, dim = x.size(1), x.size(2)
        out = x + positional_encoding(sequence_len, dim)

        for layer in self.layers:
            out = layer(x, y)

        return torch.softmax(self.linear(out), dim=-1)
