import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "prajjwal1/bert-tiny"

TRAIN_SIZE = 2_000
VALIDATION_SIZE = 500
SEED = 42

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

OUTPUT_PATH = Path("results/assets/metrics/transformer_baseline.json")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_fixed_subset(dataset: Dataset, size: int, seed: int):
    if size <= 0:
        raise ValueError("subset size must be greater than 0")

    if size > len(dataset):
        raise ValueError(f"subset size {size} exceeds dataset size {len(dataset)}")

    return dataset.shuffle(seed=seed).select(range(size))


def prepare_sst2_datasets():
    dataset = load_dataset("nyu-mll/glue", "sst2")

    # Keep the baseline small enough to run quickly on CPU.
    return DatasetDict(
        train=select_fixed_subset(dataset["train"], TRAIN_SIZE, SEED),
        validation=select_fixed_subset(
            dataset["validation"], VALIDATION_SIZE, SEED
        ),
    )


def tokenize_sst2(datasets, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    datasets = datasets.map(
        tokenize,
        batched=True,
        remove_columns=["sentence", "idx"],
    )

    datasets = datasets.rename_column("label", "labels")
    datasets.set_format("torch")

    return datasets


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            labels = batch["labels"]

            total_loss += outputs.loss.item() * labels.size(0)
            correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_baseline():
    set_seed(SEED)

    datasets = prepare_sst2_datasets()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = tokenize_sst2(datasets, tokenizer, MAX_LENGTH)

    def collate(rows):
        return tokenizer.pad(rows, return_tensors="pt")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=BATCH_SIZE,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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
        "experiment": "transformer_baseline",
        "model_name": MODEL_NAME,
        "dataset": "nyu-mll/glue:sst2",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "seed": SEED,
        "device": str(device),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        },
        "history": history,
        "final_validation_loss": history[-1]["validation_loss"],
        "final_validation_accuracy": history[-1]["validation_accuracy"],
    }


def main():
    metrics = train_baseline()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()