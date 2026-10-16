import json
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from examples.transformer_example import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    evaluate,
    make_loader,
    prepare_sst2_datasets,
    set_seed,
    tokenize_sst2,
)
from lora_morph import inject_lora

RANK = 8
ALPHA = 16
TARGET_MODULES = {"query", "value"}
OUTPUT_PATH = Path("results/assets/metrics/transformer_lora.json")


def add_lora_adapters(model, rank=RANK, alpha=ALPHA):
    inject_lora(model, TARGET_MODULES, r=rank, alpha=alpha)

    # The base model is frozen by inject_lora. Keep the task head trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def count_parameters(model):
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def train_lora():
    set_seed(SEED)

    datasets = prepare_sst2_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = tokenize_sst2(datasets, tokenizer, MAX_LENGTH)

    train_loader = make_loader(
        datasets["train"], BATCH_SIZE, tokenizer, shuffle=True, seed=SEED
    )
    validation_loader = make_loader(
        datasets["validation"], BATCH_SIZE, tokenizer, shuffle=False, seed=SEED
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model = add_lora_adapters(model).to(device)

    trainable_parameters, total_parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_examples = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / total_examples
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val acc: {validation_accuracy:.4f}"
        )

    return {
        "experiment": "transformer_lora",
        "model_name": MODEL_NAME,
        "dataset": "nyu-mll/glue:sst2",
        "train_size": 2_000,
        "validation_size": 500,
        "seed": SEED,
        "device": str(device),
        "target_modules": sorted(TARGET_MODULES),
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "package_versions": {
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "history": history,
        "final_validation_loss": history[-1]["validation_loss"],
        "final_validation_accuracy": history[-1]["validation_accuracy"],
    }


def main():
    metrics = train_lora()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
