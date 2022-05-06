import torch
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.output_dim = self.input_dim * self.num_heads

        # query - key - value
        self.qkv = torch.nn.Linear(self.input_dim, 3 * self.output_dim, bias=False)

        # linear layer
        self.linear = torch.nn.Linear(self.output_dim, self.input_dim)

    def forward(self, sequence):
        qkv = self.qkv(sequence)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attention_score = self._scaled_dot_product_attention(q, k, v)
        result = self.linear(attention_score)

        return result

    def _scaled_dot_product_attention(self, q, k, v):
        qk = q.bmm(k.transpose(1, 2))
        scale = k.size(-1) ** 0.5
        softmax = F.softmax(qk / scale, dim=-1)

        return softmax.bmm(v)


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        # normalization layers
        self.pre_lnorm = torch.nn.LayerNorm(self.input_dim)  # input normalization
        self.lnorm = torch.nn.LayerNorm(self.input_dim)  # hidden normalization

        # multihead attention
        self.multihead_attention = MultiHeadAttention(
            input_dim=self.input_dim, num_heads=self.num_heads
        )

        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.input_dim, self.input_dim),
        )

    def forward(self, input_):
        result = self.pre_lnorm(input_)
        result = self.multihead_attention(result)

        # add residual connection
        result += input_

        # pass through another layer normalization
        result = self.lnorm(result)

        # MLP and residual connection
        result = result + self.mlp(result)

        return result


class ViT(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        patch_shape=(16, 16),
        hidden_dim=8,
        num_encoders=1,
        num_heads=2,
        output_dim=10,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoders = num_encoders

        self.ver_patches = self.input_shape[1] / self.patch_shape[0]
        self.hor_patches = self.input_shape[2] / self.patch_shape[1]

        self.num_patches = int(self.ver_patches * self.hor_patches)
        self.dim_patch = self.patch_shape[0] * self.patch_shape[1]
        self.dim = int(self.input_shape[0] * self.dim_patch)

        # linear patch mapper
        self.linear = torch.nn.Linear(self.dim, self.hidden_dim)

        # classification token
        self.class_token = torch.nn.Parameter(torch.rand(1, self.hidden_dim))

        # encoder blocks
        self.encoders = torch.nn.ModuleList([
            Encoder(input_dim=self.hidden_dim, num_heads=self.num_heads)
            for _ in range(self.num_encoders)
        ])

        # classification layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, output_dim), torch.nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # divide images into patches
        n, c, w, h = images.shape
        patches = images.reshape(n, self.num_patches, self.dim)

        # tokenize image patches
        tokens = self.linear(patches)

        # append the class token
        tokens = torch.stack(
            [torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))]
        )

        # add positional encoding to each token
        tokens += self._get_positional_encodings(self.num_patches + 1, self.hidden_dim).repeat(
            n, 1, 1
        )

        # pass through encoders
        result = tokens
        for encoder in self.encoders:
            result = encoder(result)

        # get the classification token
        result = result[:, 0]

        return self.mlp(result)

    def _get_positional_encodings(self, sequence_length, dim):
        result = torch.ones(sequence_length, dim)

        for i in range(sequence_length):
            for j in range(dim):
                if j % 2 == 0:
                    result[i][j] = np.sin(i / (10_000 ** (j / dim)))
                else:
                    result[i][j] = np.cos(i / (10_000 ** ((j - 1) / dim)))

        return result


m = ViT(input_shape=(3, 28, 28), patch_shape=(4, 4))
x = torch.rand(3, 3, 28, 28)

print(m(x).shape)
