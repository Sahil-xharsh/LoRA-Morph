import argparse
import hashlib
import json
from pathlib import Path

import torch
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
    SEED,
    evaluate,
    make_loader,
    prepare_sst2_datasets,
    set_seed,
    tokenize_sst2,
)

OUTPUT_PATH = Path("results/assets/metrics/accuracy_diagnosis.json")


# Dataset Reports
def dataset_report(raw_datasets, tokenizer):
    """Return labels, token lengths and truncation statistics."""
    label_counts = {}
    lengths = []
    truncated = 0
    for split in ("train", "validation"):
        for row in raw_datasets[split]:
            label = int(row["label"])
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1
            full_length = len(tokenizer(row["sentence"], truncation=False)["input_ids"])
            kept_length = len(
                tokenizer(row["sentence"], truncation=True, max_length=MAX_LENGTH)[
                    "input_ids"
                ]
            )
            lengths.append(kept_length)
            truncated += int(full_length > kept_length)
    return {
        "label_mapping": {"0": "negative", "1": "positive"},
        "label_counts": label_counts,
        "token_length_min": min(lengths),
        "token_length_max": max(lengths),
        "token_length_mean": sum(lengths) / len(lengths),
        "truncated_examples": truncated,
        "max_length": MAX_LENGTH,
    }


# Model Parameters
def checksum(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_norms(model, predicate):
    return {
        name: float(parameter.detach().norm().cpu())
        for name, parameter in model.named_parameters()
        if predicate(name, parameter)
    }

# runs diagnostic condition and returns MetaData
def train_or_evaluate(kind, datasets, tokenizer, learning_rate):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    if kind == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif kind == "lora":
        model = add_lora_adapters(model, rank=8, alpha=ALPHA)
    model = model.to(device)

    train_loader = make_loader(
        datasets["train"], BATCH_SIZE, tokenizer, shuffle=True, seed=SEED
    )
    validation_loader = make_loader(
        datasets["validation"], BATCH_SIZE, tokenizer, shuffle=False, seed=SEED
    )
    initial_checksum = checksum(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate) if trainable else None
    initial_lora_norms = parameter_norms(
        model, lambda name, _: ".A" in name or ".B" in name
    )
    max_gradient_norm = 0.0
    gradient_steps = 0
    update_steps = 0

    if kind != "untrained":
        for _ in range(EPOCHS):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                loss = model(**batch).loss
                loss.backward()
                for parameter in trainable:
                    if parameter.grad is not None:
                        gradient_steps += 1
                        max_gradient_norm = max(
                            max_gradient_norm,
                            float(parameter.grad.detach().norm().cpu()),
                        )
                optimizer.step()
                update_steps += 1

    validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
    final_lora_norms = parameter_norms(
        model, lambda name, _: ".A" in name or ".B" in name
    )
    return {
        "condition": kind,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "learning_rate": learning_rate,
        "epochs": 0 if kind == "untrained" else EPOCHS,
        "update_steps": update_steps,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "initial_lora_norms": initial_lora_norms,
        "final_lora_norms": final_lora_norms,
        "max_gradient_norm": max_gradient_norm,
        "gradient_observations": gradient_steps,
        "lora_parameters_changed": (
            initial_checksum != checksum(model) if kind == "lora" else False
        ),
        "initial_parameter_checksum": initial_checksum,
        "final_parameter_checksum": checksum(model),
    }


def run_diagnosis(learning_rate):
    set_seed(SEED)
    raw_datasets = prepare_sst2_datasets()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = tokenize_sst2(raw_datasets, tokenizer, MAX_LENGTH)
    report = {
        "experiment": "sst2_accuracy_diagnosis",
        "model_name": MODEL_NAME,
        "seed": SEED,
        "target_modules": sorted(TARGET_MODULES),
        "learning_rate": learning_rate,
        "dataset": dataset_report(raw_datasets, tokenizer),
        "runs": [],
    }
    for repeat in (1, 2):
        for kind in ("untrained", "head_only", "lora"):
            result = train_or_evaluate(kind, datasets, tokenizer, learning_rate)
            result["repeat"] = repeat
            report["runs"].append(result)
    report["deterministic_repeats_match"] = all(
        report["runs"][index]["final_parameter_checksum"]
        == report["runs"][index + 3]["final_parameter_checksum"]
        for index in range(3)
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    report = run_diagnosis(args.learning_rate)
    for run in report["runs"][:3]:
        print(
            f"{run['condition']}: accuracy={run['validation_accuracy']:.4f} "
            f"loss={run['validation_loss']:.4f} updates={run['update_steps']}"
        )
    print(f"deterministic_repeats_match={report['deterministic_repeats_match']}")
    print(f"Metrics written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
