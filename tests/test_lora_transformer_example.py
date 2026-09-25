import torch
from transformers import BertConfig, BertForSequenceClassification

from examples.lora_transformer_example import add_lora_adapters, count_parameters
from lora_morph import LoRALinear


def test_transformer_lora_targets_query_and_value_projections():
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=2,
        )
    )

    add_lora_adapters(model, rank=4, alpha=8)

    lora_modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    assert len(lora_modules) == 4
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_transformer_lora_only_trains_adapters_and_classifier():
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=2,
        )
    )

    add_lora_adapters(model, rank=4, alpha=8)
    trainable, total = count_parameters(model)

    assert 0 < trainable < total
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".A" in name or ".B" in name or name.startswith("classifier.")
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not (".A" in name or ".B" in name or name.startswith("classifier."))
    )


def test_transformer_lora_forward_has_expected_shape():
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=2,
        )
    )
    add_lora_adapters(model, rank=4, alpha=8)

    output = model(
        input_ids=torch.randint(0, 100, (2, 8)),
        attention_mask=torch.ones(2, 8, dtype=torch.long),
        labels=torch.tensor([0, 1]),
    )

    assert output.logits.shape == (2, 2)
    assert output.loss.ndim == 0
