import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be greater than 0")

        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base_linear.weight.requires_grad = False

        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(base_linear.in_features, r) * 0.01
        )
        self.B = nn.Parameter(
            torch.zeros(r, base_linear.out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_linear(x)
        lora_output = x @ self.A @ self.B

        return base_output + self.scaling * lora_output