import torch


class Residual(torch.nn.Module):
    def __init__(self, sublayer, dim, dropout=0.1) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.dim = dim
        self.dropout = dropout

        self.norm_layer = torch.nn.LayerNorm(dim)
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, *tensors):
        """
        Assume that the ordering of the tensors is as follows:
        1. query
        2. key
        3. value
        """

        return self.norm_layer(tensors[0] + self.dropout_layer(self.sublayer(*tensors)))


class FeedForward(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=512) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self.output_dim = output_dim

        self.input_layer = torch.nn.Linear(self.input_dim, self.hidden_dim[0])
        self.output_layer = torch.nn.Linear(self.hidden_dim[-1], self.output_dim)

        self.hidden_layers = []
        if len(self.hidden_dim) > 1:
            for fan_in, fan_out in zip(self.hidden_dim[:-1], self.hidden_dim[1:]):
                self.hidden_layers.append((torch.nn.Linear(fan_in, fan_out), torch.nn.ReLU()))

        self.net = torch.nn.Sequential(
            self.input_layer,
            torch.nn.ReLU(),
            *[l for hidden in self.hidden_layers for l in hidden],
            self.output_layer
        )

    def forward(self, x):
        return self.net(x)


# x = [1, 2, 3, 4]
# y = []
# for fan_in, fan_out in zip(x[:-1], x[1:]):
#     y.append((fan_in, fan_out))

# print(*[a for i in y for a in i])

# ff = FeedForward(hidden_dim=[512, 512])
# t = torch.rand(17876, 16, 512)
# ff(t)
