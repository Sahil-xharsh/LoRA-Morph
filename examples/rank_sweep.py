import json
from importlib.metadata import version
from pathlib import Path

import torch
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
    make_loader,
    prepare_sst2_datasets,
    set_seed,
    tokenize_sst2,
)
from lora_morph import LoRATracker

RANKS = (2, 4, 8, 16)
OUTPUT_PATH = Path("results/assets/metrics/rank_sweep.json")
CHECKPOINT_DIR = Path("results/assets/checkpoints")
PLOT_DIR = Path("results/assets/plots")


def train_rank(rank, datasets, tokenizer, device):
    """Train one rank from a fresh, deterministic model and loader state."""
    # Reset the model RNG for every rank so the pretrained and classifier
    # initialization is comparable across ranks. Only the adapter shape varies.
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = add_lora_adapters(model, rank=rank, alpha=ALPHA).to(device)
    initial_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    train_loader = make_loader(
        datasets["train"], BATCH_SIZE, tokenizer, shuffle=True, seed=SEED + rank
    )
    validation_loader = make_loader(
        datasets["validation"], BATCH_SIZE, tokenizer, shuffle=False, seed=SEED
    )
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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "base_model": MODEL_NAME,
            "config": model.config.to_dict(),
            "rank": rank,
            "alpha": ALPHA,
            "target_modules": sorted(TARGET_MODULES),
            "initial_state_dict": initial_state,
            "state_dict": adapter_state,
        },
        CHECKPOINT_DIR / f"rank_{rank}.pt",
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [train_rank(rank, datasets, tokenizer, device) for rank in RANKS]
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
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    points = [
        (item["rank"], item["final_validation_accuracy"]) for item in results["results"]
    ]
    max_accuracy = max(value for _, value in points)
    bars = []
    for index, (rank, accuracy) in enumerate(points):
        height = 160 * accuracy / max_accuracy
        x = 70 + index * 100
        y = 220 - height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="50" height="{height:.1f}" fill="#4c78a8" />'
            f'<text x="{x + 25}" y="245" text-anchor="middle">r={rank}</text>'
            f'<text x="{x + 25}" y="{y - 6:.1f}" text-anchor="middle">{accuracy:.3f}</text>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="280" '
        'viewBox="0 0 500 280"><rect width="100%" height="100%" fill="white"/>'
        '<text x="250" y="25" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">SST-2 LoRA rank sweep</text>' + "".join(bars) + "</svg>\n"
    )
    (PLOT_DIR / "rank_sweep.svg").write_text(svg, encoding="utf-8")
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
