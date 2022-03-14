import torch
import torch.nn.functional as F

from models.layers.misc import Residual, FeedForward
from models.transformer.multi_head_attention import MultiHeadAttentionLayer


def positional_encoding(sequence_len, dim, device=torch.device('cpu')):
    pos = torch.arange(sequence_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim_ = torch.arange(dim, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim_ // dim))

    return torch.where(dim_.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class TransformerEncoderLayer(torch.nn.Module):
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

        # attention layer
        self.attention_layer = Residual(
            MultiHeadAttentionLayer(
                self.num_heads, self.input_dim, self.query_dim, self.key_dim, self.value_dim
            ),
            self.input_dim,
            self.dropout,
        )

        # feed forward network
        self.feed_forward_layer = Residual(
            FeedForward(
                input_dim=self.input_dim, hidden_dim=self.feedforward_dim, output_dim=self.output_dim
            ),
            self.input_dim,
            self.dropout,
        )

    def forward(self, x):
        x = self.attention_layer(x, x, x)
        return self.feed_forward_layer(x)


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers=6,
        num_heads=8,
        input_dim=512,
        output_dim=512,
        vocab_size=None,
        feedforward_dim=2048,
        dropout=0.1,
        device=torch.device('cpu')
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.device = device

        # layers
        self.embedding = torch.nn.Embedding(vocab_size, input_dim)

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_heads=self.num_heads,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    feedforward_dim=self.feedforward_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        x = self.embedding(x)
        sequence_len, dim = x.size(1), x.size(2)
        x += positional_encoding(sequence_len, dim, device=self.device)

        for layer in self.layers:
            x = layer(x)

        return x
