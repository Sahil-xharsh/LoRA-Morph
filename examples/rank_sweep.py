import json
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from examples.lora_transformer_example import (
    ALPHA,
    TARGET_MODULES,
    add_lora_adapters,
    count_parameters,
)
from examples.transformer_example import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
    evaluate,
    prepare_sst2_datasets,
    set_seed,
    tokenize_sst2,
)
from lora_morph import LoRATracker


RANKS = (2, 4, 8, 16)
OUTPUT_PATH = Path("results/assets/metrics/rank_sweep.json")


def train_rank(rank, train_loader, validation_loader, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model = add_lora_adapters(model, rank=rank, alpha=ALPHA).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    tracker = LoRATracker()
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
            tracker.record_loss(loss.item())
            tracker.record_adapters(model)
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

    trainable, total = count_parameters(model)
    return {
        "rank": rank,
        "alpha": ALPHA,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "history": history,
        "tracker": tracker.to_dict(),
        "final_validation_accuracy": history[-1]["validation_accuracy"],
    }


def run_sweep():
    set_seed(SEED)
    raw_datasets = prepare_sst2_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = tokenize_sst2(raw_datasets, tokenizer, MAX_LENGTH)

    def collate(rows):
        return tokenizer.pad(rows, return_tensors="pt")

    train_loader = DataLoader(
        datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        datasets["validation"], batch_size=BATCH_SIZE, collate_fn=collate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        train_rank(rank, train_loader, validation_loader, device) for rank in RANKS
    ]
    return {
        "experiment": "rank_sweep",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "target_modules": sorted(TARGET_MODULES),
        "device": str(device),
        "package_versions": {
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
        "results": results,
    }


def main():
    results = run_sweep()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
