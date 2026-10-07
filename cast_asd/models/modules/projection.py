import torch.nn as nn


class LinearHead(nn.Module):
    """LayerNorm -> Linear, with the Linear initialised to N(0, 0.02^2)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.norm(x))

