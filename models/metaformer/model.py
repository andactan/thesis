from multiprocessing import Pool
import torch


class Pooling(torch.nn.Module):
    """Average pooling layer for 1D inputs

    Args:
        pool_size (int): kernel size of average pooling
    """
    def __init__(self, pool_size=3) -> None:
        super().__init__()
        self.pool_size = pool_size

        # to resemble the input-output size equality in attention
        # add padding and set the stride to '1'
        self.pool = torch.nn.AvgPool2d(
            kernel_size=self.pool_size, stride=1, padding=self.pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        # this approach works slightly better than
        # the one without subtraction
        # more details: https://github.com/sail-sg/poolformer/issues/4
        return self.pool(x) - x

x = torch.ones(1, 1, 5)
p = Pooling()
print(p(x).size())


class PatchEmbed(torch.nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        patch_size = patch_size
        stride = stride
        padding = padding
        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


x = torch.ones(1, 1, 512, 512)
embed = PatchEmbed(in_chans=1)
embedding = embed(x)
pooled = p(embedding)
print(pooled.size())
