import torch

from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder

class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=6,
        input_dim=512,
        output_dim=512,
        feedforward_dim=2048,
        dropout=0.1
    ) -> None:
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.encoder = TransformerEncoder(
            num_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=self.num_decoder_layers,
            num_heads=self.num_heads,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout
        )

    def forward(self, target, source):
        return self.decoder(target, self.encoder(source))