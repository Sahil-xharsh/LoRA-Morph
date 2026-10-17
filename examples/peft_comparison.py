import copy
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from examples.lora_transformer_example import ALPHA, TARGET_MODULES, add_lora_adapters
from examples.transformer_example import (
    BATCH_SIZE,
    EPOCHS,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    evaluate,
    prepare_sst2_datasets,
    set_seed,
    tokenize_sst2,
)
from lora_morph import LoRALinear

RANK = 8
LEARNING_RATE = 1e-4
OUTPUT_PATH = Path("results/assets/metrics/peft_comparison.json")


def add_peft_adapters(model):
    config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=sorted(TARGET_MODULES),
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, config)


def copy_custom_weights_to_peft(custom_model, peft_model):
    with torch.no_grad():
        for name, module in custom_model.named_modules():
            if not isinstance(module, LoRALinear):
                continue

            peft_layer = peft_model.get_submodule(name)
            peft_layer.lora_A["default"].weight.copy_(module.A.T)
            peft_layer.lora_B["default"].weight.copy_(module.B.T)


def forward_deviation(custom_model, peft_model, inputs):
    custom_model.eval()
    peft_model.eval()

    with torch.no_grad():
        custom_logits = custom_model(**inputs).logits
        peft_logits = peft_model(**inputs).logits

    return (custom_logits - peft_logits).abs().max().item()


def adapter_gradient_cosine(custom_model, peft_model):
    similarities = []

    for name, module in custom_model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        peft_layer = peft_model.get_submodule(name)
        peft_a_grad = peft_layer.lora_A["default"].weight.grad
        peft_b_grad = peft_layer.lora_B["default"].weight.grad
        pairs = (
            (module.A.grad, peft_a_grad.T if peft_a_grad is not None else None),
            (module.B.grad, peft_b_grad.T if peft_b_grad is not None else None),
        )

        for custom_grad, peft_grad in pairs:
            if custom_grad is None or peft_grad is None:
                continue

            custom_grad = custom_grad.reshape(1, -1)
            peft_grad = peft_grad.reshape(1, -1)
            if custom_grad.norm() == 0 and peft_grad.norm() == 0:
                similarities.append(1.0)
            else:
                similarities.append(
                    F.cosine_similarity(custom_grad, peft_grad).item()
                )

    return sum(similarities) / len(similarities) if similarities else None
