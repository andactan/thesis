import torch


class Compression(torch.nn.Module):
    def __init__(self, compression_rate: int, dim_model: int) -> None:
        """Simple wrapper for compression operation

        Args:
            compression_rate (int): compression rate
            dim_model (int): embedding size
        """
        super().__init__()
        self.compression_rate = compression_rate
        self.dim_model = dim_model

        self.conv = torch.nn.Conv1d(
            dim_model, dim_model, kernel_size=self.dim_model, stride=self.compression_rate
        )

    def forward(self, mem: torch.Tensor):
        # permute the dimensions of mem, since conv layer accepts the form
        # [batch, features, sequence]
        # although mem is in a form of [sequence, batch, features]
        mem = mem.permute(1, 2, 0)

        # get the compressed memory
        c_mem = self.conv(mem)

        # permute back to [sequence, batch, features]
        return c_mem.permute(2, 0, 1)