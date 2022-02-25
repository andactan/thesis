import torch


def positional_encoding(sequence_len, dim, device=torch.device("cpu")):
    pos = torch.arange(sequence_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim_ = torch.arange(dim, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim_ // dim))

    return torch.where(dim_.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
