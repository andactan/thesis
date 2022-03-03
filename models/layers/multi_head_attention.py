import torch
import torch.nn.functional as F


class AttentionHead(torch.nn.Module):
    def __init__(self, input_dim, query_dim, key_dim, value_dim) -> None:
        super().__init__()
        self.q = torch.nn.Linear(input_dim, query_dim, bias=False)
        self.k = torch.nn.Linear(input_dim, key_dim, bias=False)
        self.v = torch.nn.Linear(input_dim, value_dim, bias=False)

    def forward(self, query, key, value):
        query_ = self.q(query)
        key_ = self.k(key)
        value_ = self.v(value)

        return self._scaled_dot_product(query_, key_, value_)

    def _get_scores(self, query, key):
        qk = query.bmm(key.transpose(1, 2))
        scale = key.size(-1) ** 0.5
        softmax = F.softmax(qk / scale, dim=-1)

        return softmax

    def _scaled_dot_product(self, query, key, value):
        softmax = self._get_scores(query, key)

        return softmax.bmm(value)


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, num_heads, input_dim, query_dim, key_dim, value_dim) -> None:
        super().__init__()

        # define configurations
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # attention heads
        self.heads = torch.nn.ModuleList(
            [AttentionHead(input_dim, query_dim, key_dim, value_dim) for _ in range(self.num_heads)]
        )

        # linear layer
        self.linear = torch.nn.Linear(self.num_heads * self.value_dim, input_dim)

    def forward(self, query, key, value):
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
