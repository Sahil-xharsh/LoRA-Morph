
import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)

from lora_morph import inject_lora

MODEL_NAME = "prajjwal1/bert-tiny"
TARGET_MODULES = {"query", "value"}
TASK = "sst2-sentiment"
METRICS_PATH = Path("results/assets/metrics/rank_sweep.json")
PLOT_DIR = Path("results/assets/plots")


def adapter_state_from_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load and normalize trainable adapter tensors from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
        if ".A" in name or ".B" in name or name.startswith("classifier.")
    }


def load_checkpoint(checkpoint_path: Path):
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)

# Checkpoint fall Back
def build_model(checkpoint, rank: int, alpha: float):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
    except Exception as error:
        if "initial_state_dict" not in checkpoint or "config" not in checkpoint:
            raise RuntimeError(
                "base model is unavailable and checkpoint has no fallback state"
            ) from error
        config = BertConfig.from_dict(checkpoint["config"])
        model = BertForSequenceClassification(config)
        model.load_state_dict(checkpoint["initial_state_dict"], strict=True)
    inject_lora(model, TARGET_MODULES, r=rank, alpha=alpha)
    return model


def best_validation_accuracy(rank: int) -> float:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for result in metrics["results"]:
        if result["rank"] == rank:
            return float(result["history"][-1]["validation_accuracy"])
    raise ValueError(f"rank {rank} is not present in {METRICS_PATH}")


def verify_roundtrip(
    adapter_path: Path,
    checkpoint,
    state: dict[str, torch.Tensor],
    rank: int,
    alpha: float,
):
    torch.manual_seed(42)
    original = build_model(checkpoint, rank, alpha)
    original.load_state_dict(state, strict=False)
    original.eval()

    reloaded = build_model(checkpoint, rank, alpha)
    reloaded.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=False)
    reloaded.eval()
    inputs = {
        "input_ids": torch.tensor(
            [[101, 2023, 2003, 1037, 7099, 102], [101, 2023, 2003, 2204, 102, 0]]
        ),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]),
    }
    with torch.no_grad():
        original_logits = original(**inputs).logits
        reloaded_logits = reloaded(**inputs).logits
    maximum_error = float((original_logits - reloaded_logits).abs().max())
    if maximum_error > 1e-5:
        raise RuntimeError(f"adapter roundtrip mismatch: max error {maximum_error}")
    return maximum_error

#Adapter weights and metadata
def export_bundle(
    checkpoint_path: Path, rank: int, alpha: float, output_dir: Path
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    state = adapter_state_from_checkpoint(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter.safetensors"
    save_file(state, str(adapter_path), metadata={"format": "pt"})
    config = {
        "base_model": MODEL_NAME,
        "r": rank,
        "alpha": alpha,
        "target_modules": sorted(TARGET_MODULES),
        "task": TASK,
        "seed": 42,
        "train_examples": 2000,
        "val_examples": 500,
        "lora_morph_version": version("lora-morph"),
        "final_val_accuracy": best_validation_accuracy(rank),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    for plot in PLOT_DIR.glob("rank_sweep.*"):
        shutil.copy2(plot, output_dir / plot.name)
    roundtrip_error = verify_roundtrip(adapter_path, checkpoint, state, rank, alpha)
    print(f"roundtrip_max_logit_error={roundtrip_error:.8g}")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, default=Path("hf_release"))
    args = parser.parse_args()
    export_bundle(args.checkpoint, args.rank, args.alpha, args.out)
    print(f"Exported adapter bundle to {args.out}")


if __name__ == "__main__":
    main()
