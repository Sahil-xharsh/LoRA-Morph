import torch

from examples.cnn_example import FashionCNN, add_lora, count_parameters
from lora_morph import LoRALinear


def test_cnn_lora_replaces_only_linear_classifier_layers():
    model = add_lora(FashionCNN())

    assert isinstance(model.fc1, LoRALinear)
    assert isinstance(model.fc2, LoRALinear)
    assert isinstance(model.features[0], torch.nn.Conv2d)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())


def test_cnn_lora_keeps_only_adapters_trainable():
    model = add_lora(FashionCNN())
    trainable, total = count_parameters(model)

    assert 0 < trainable < total
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".A" in name or ".B" in name
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "base_linear" in name or name.startswith("features.")
    )


def test_cnn_lora_forward_has_ten_class_outputs():
    model = add_lora(FashionCNN())

    output = model(torch.randn(3, 1, 28, 28))

    assert output.shape == (3, 10)
