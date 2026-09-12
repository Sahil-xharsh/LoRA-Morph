from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from lora_morph import LoRALinear, inject_lora

# torch.Tensor
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


def test_lora_linear_applies_scaled_low_rank_update():
    base_linear = nn.Linear(3, 2, bias=False)
    lora_linear = LoRALinear(base_linear, r=2, alpha=4)
    x = torch.tensor([[1.0, 2.0, 3.0]])

    with torch.no_grad():
        lora_linear.A.copy_(
            torch.tensor([[1.0, 0.0],[0.0, 1.0], [1.0, 1.0]])
        )
        lora_linear.B.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))

    expected_update = (4 / 2) * (x @ lora_linear.A @ lora_linear.B) 

    assert torch.allclose(
        lora_linear(x),
        base_linear(x) + expected_update,
    )


def test_lora_linear_supports_layers_without_bias():
    lora_linear = LoRALinear(nn.Linear(4, 3, bias=False), r=2, alpha=2)

    assert lora_linear.base_linear.bias is None
    assert lora_linear(x=torch.randn(5, 4)).shape == (5, 3)


def test_lora_linear_rejects_non_positive_rank():
    base_linear = nn.Linear(4, 3)

    for rank in (0, -1):
        with pytest.raises(ValueError, match="^r must be greater than 0$"):
            LoRALinear(base_linear, r=rank, alpha=2)


def test_inject_lora_replaces_only_targeted_linear_layers():
    model = nn.Sequential(
        OrderedDict(
            [
                ("input_projection", nn.Linear(4, 4)),
                ("activation", nn.ReLU()),
                ("output_projection", nn.Linear(4, 2)),
            ]
        )
    )

    injected_model = inject_lora(
        model,
        target_module_names={"input_projection"},
        r=2,
        alpha=4,
    )

    assert injected_model is model
    assert isinstance(model.input_projection, LoRALinear)
    assert isinstance(model.output_projection, nn.Linear)


def test_inject_lora_freezes_base_parameters_and_trains_adapters():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))

    inject_lora(model, target_module_names={"0", "1"}, r=2, alpha=2)

    trainable_parameters = [  # type: list[nn.Parameter]
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) == 4
    assert all(parameter.requires_grad for parameter in trainable_parameters)
    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) not in trainable_ids
    )


def test_inject_lora_supports_forward_and_backward():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    inject_lora(model, target_module_names={"0", "2"}, r=2, alpha=2)

    loss = model(torch.randn(5, 4)).sum()
    loss.backward()

    assert model[0].A.grad is not None
    assert model[0].B.grad is not None
    assert model[2].A.grad is not None
    assert model[2].B.grad is not None
    assert model[0].base_linear.weight.grad is None
    assert model[2].base_linear.weight.grad is None


def test_inject_lora_rejects_unmatched_target_names():
    with pytest.raises(ValueError, match="No matching nn.Linear modules found"):
        inject_lora(nn.Sequential(nn.Linear(4, 2)), {"missing"}, r=2, alpha=2)
