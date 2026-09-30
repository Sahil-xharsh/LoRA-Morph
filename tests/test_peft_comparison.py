import copy

import torch
from transformers import BertConfig, BertForSequenceClassification

from examples.peft_comparison import (
    add_peft_adapters,
    add_lora_adapters,
    copy_custom_weights_to_peft,
    forward_deviation,
)


def test_custom_and_peft_forward_outputs_match_after_weight_copy():
    base_model = BertForSequenceClassification(
        BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=2,
        )
    )
    custom_model = add_lora_adapters(copy.deepcopy(base_model), rank=2, alpha=4)
    peft_model = add_peft_adapters(copy.deepcopy(base_model), rank=2, alpha=4)
    copy_custom_weights_to_peft(custom_model, peft_model)

    inputs = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }

    assert forward_deviation(custom_model, peft_model, inputs) <= 1e-5
