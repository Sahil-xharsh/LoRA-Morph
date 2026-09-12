from collections.abc import Iterable

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be greater than 0")

        self.base_linear = base_linear  # type: nn.Linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.base_linear.weight.requires_grad = False

        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(base_linear.in_features, r) * 0.01
        )  # type: nn.Parameter
        self.B = nn.Parameter(
            torch.zeros(r, base_linear.out_features)
        )  # type: nn.Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_linear(x)
        lora_output = x @ self.A @ self.B

        return base_output + self.scaling * lora_output



def inject_lora(
    model: nn.Module,
    target_module_names: Iterable[str],
    r: int,
    alpha: float,
) -> nn.Module:
    """Replaced matching linear layers, keeping rest fixed """
    targets = set(target_module_names)  # type: set[str]

    for param in model.parameters():
        param.requires_grad = False

    matches = []  # type: list[tuple[str, nn.Linear]]
    for full_name, module in model.named_modules():
        name = full_name.rsplit(".", 1)[-1]
        if full_name and isinstance(module, nn.Linear) and name in targets:
            matches.append((full_name, module))

    if not matches:
        names = ", ".join(sorted(targets)) or "<none>"
        raise ValueError(f"No matching nn.Linear modules found for: {names}")

    for full_name, base in matches:
        if "." in full_name:
            parent_name, name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, name = model, full_name

        setattr(parent, name, LoRALinear(base, r=r, alpha=alpha))

    return model
