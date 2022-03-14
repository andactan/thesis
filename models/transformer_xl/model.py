import torch

from models.layers.relative_multi_head_attention import RelativeAttentionHead
class RelativeDecoderLayer(torch.nn.Module):
    def __init__(self, num_head, dim_model, dim_head, dim_inner, dropout, **kwargs) -> None:
        super().__init__()

        self.attention = RelativeAttentionHead()