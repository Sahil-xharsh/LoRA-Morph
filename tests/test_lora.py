import torch
import torch.nn as nn

from lora_morph.lora import LoRALinear


def test_lora_linear_starts_as_exact_noop():
    torch.manual_seed(42)

    base_linear = nn.Linear(10, 10)
    lora_linear = LoRALinear(
        base_linear,
        r=4,
        alpha=8,
    )

    x = torch.randn(5, 10)

    base_output = base_linear(x)
    lora_output = lora_linear(x)

    assert lora_output.shape == base_output.shape
    assert torch.equal(lora_output, base_output)

    assert base_linear.weight.requires_grad is False
    assert base_linear.bias.requires_grad is False

    assert lora_linear.A.requires_grad is True
    assert lora_linear.B.requires_grad is True