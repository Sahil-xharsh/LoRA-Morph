import copy
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from examples.lora_transformer_example import (
    ALPHA,
    TARGET_MODULES,
    add_lora_adapters,
)
from examples.transformer_example import (
    BATCH_SIZE,
    EPOCHS,
    MAX_LENGTH,
    MODEL_NAME,
    evaluate,
    prepare_sst2_datasets,
    SEED,
    set_seed,
    tokenize_sst2,
)
from lora_morph import LoRALinear


RANK = 8
COMPARISON_LEARNING_RATE = 1e-4
OUTPUT_PATH = Path("results/assets/metrics/peft_comparison.json")


def add_peft_adapters(model, rank=RANK, alpha=ALPHA):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=sorted(TARGET_MODULES),
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, config)


def copy_custom_weights_to_peft(custom_model, peft_model):
    for name, module in custom_model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        peft_module = peft_model.get_submodule(name)
        peft_module.lora_A["default"].weight.data.copy_(module.A.data.T)
        peft_module.lora_B["default"].weight.data.copy_(module.B.data.T)


def forward_deviation(custom_model, peft_model, inputs):
    custom_model.eval()
    peft_model.eval()
    with torch.no_grad():
        custom_logits = custom_model(**inputs).logits
        peft_logits = peft_model(**inputs).logits
    return (custom_logits - peft_logits).abs().max().item()


def adapter_gradient_cosine(custom_model, peft_model):
    cosines = []
    for name, module in custom_model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        peft_module = peft_model.get_submodule(name)
        custom_gradients = [module.A.grad.flatten(), module.B.grad.flatten()]
        peft_gradients = [
            peft_module.lora_A["default"].weight.grad.T.flatten(),
            peft_module.lora_B["default"].weight.grad.T.flatten(),
        ]
        for custom_gradient, peft_gradient in zip(custom_gradients, peft_gradients):
            if custom_gradient.norm() == 0 and peft_gradient.norm() == 0:
                cosines.append(1.0)
            else:
                cosines.append(
                    F.cosine_similarity(
                        custom_gradient.unsqueeze(0), peft_gradient.unsqueeze(0)
                    ).item()
                )
    return sum(cosines) / len(cosines)


def train_model(model, train_loader, validation_loader, device):
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=COMPARISON_LEARNING_RATE,
    )
    history = []
    initial_loss, initial_accuracy = evaluate(model, validation_loader, device)
    history.append(
        {"epoch": 0, "validation_loss": initial_loss, "validation_accuracy": initial_accuracy}
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            model(**batch).loss.backward()
            optimizer.step()

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
    return history


def training_comparison():
    datasets = prepare_sst2_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = tokenize_sst2(datasets, tokenizer, MAX_LENGTH)

    def collate(rows):
        return tokenizer.pad(rows, return_tensors="pt")

    train_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=BATCH_SIZE, collate_fn=collate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    custom_model = add_lora_adapters(copy.deepcopy(base_model), RANK, ALPHA)
    peft_model = add_peft_adapters(copy.deepcopy(base_model), RANK, ALPHA)
    copy_custom_weights_to_peft(custom_model, peft_model)

    custom_history = train_model(
        custom_model, train_loader, validation_loader, device
    )
    peft_history = train_model(peft_model, train_loader, validation_loader, device)
    custom_losses = [item["validation_loss"] for item in custom_history]
    peft_losses = [item["validation_loss"] for item in peft_history]
    curve_rmse = math.sqrt(
        sum((custom - peft) ** 2 for custom, peft in zip(custom_losses, peft_losses))
        / len(custom_losses)
    )
    custom_accuracy = custom_history[-1]["validation_accuracy"]
    peft_accuracy = peft_history[-1]["validation_accuracy"]
    return {
        "custom_history": custom_history,
        "peft_history": peft_history,
        "curve_rmse": curve_rmse,
        "final_accuracy_difference": abs(custom_accuracy - peft_accuracy),
        "custom_loss_non_increasing_from_initial": custom_losses[-1] <= custom_losses[0],
        "peft_loss_non_increasing_from_initial": peft_losses[-1] <= peft_losses[0],
    }


def compare_initialization():
    set_seed(SEED)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    custom_model = add_lora_adapters(copy.deepcopy(base_model), RANK, ALPHA)
    peft_model = add_peft_adapters(copy.deepcopy(base_model), RANK, ALPHA)
    copy_custom_weights_to_peft(custom_model, peft_model)

    inputs = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    custom_model.eval()
    peft_model.eval()
    custom_model(**inputs).loss.backward()
    peft_model(**inputs).loss.backward()

    return {
        "max_forward_deviation": forward_deviation(
            custom_model, peft_model, inputs
        ),
        "adapter_gradient_cosine": adapter_gradient_cosine(custom_model, peft_model),
        "custom_trainable_parameters": sum(
            parameter.numel()
            for parameter in custom_model.parameters()
            if parameter.requires_grad
        ),
        "peft_trainable_parameters": sum(
            parameter.numel()
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        ),
    }


def main():
    results = {
        "experiment": "peft_comparison",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "rank": RANK,
        "alpha": ALPHA,
        "learning_rate": COMPARISON_LEARNING_RATE,
        "target_modules": sorted(TARGET_MODULES),
        "package_versions": {
            "torch": version("torch"),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "initialization_parity": compare_initialization(),
        "training_comparison": training_comparison(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
