import json
from pathlib import Path

import torch

from examples import lora_transformer_example as lte

OUTPUT_DIR = Path("hf_release")


def main() -> None:
    captured_model = {}
    original_add_lora_adapters = lte.add_lora_adapters

    def capture_model(model, *args, **kwargs):
        model = original_add_lora_adapters(model, *args, **kwargs)
        captured_model["model"] = model
        return model

    # Keep a reference to the model after the LoRA adapters are injected.
    lte.add_lora_adapters = capture_model

    try:
        metrics = lte.train_lora()
    finally:
        lte.add_lora_adapters = original_add_lora_adapters

    model = captured_model["model"]

    # Export only trainable parameters, including the LoRA adapters
    # and the classifier head.
    adapter_weights = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(adapter_weights, OUTPUT_DIR / "adapter.pt")

    config = {
        "base_model": lte.MODEL_NAME,
        "task": "sst2-sentiment",
        "num_labels": 2,
        "rank": lte.RANK,
        "alpha": lte.ALPHA,
        "target_modules": sorted(lte.TARGET_MODULES),
        "max_length": lte.MAX_LENGTH,
        "seed": lte.SEED,
        "epochs": lte.EPOCHS,
        "learning_rate": lte.LEARNING_RATE,
        "batch_size": lte.BATCH_SIZE,
    }

    for filename, data in (
        ("config.json", config),
        ("metrics.json", metrics),
    ):
        output_path = OUTPUT_DIR / filename
        output_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )

    adapter_path = OUTPUT_DIR / "adapter.pt"
    adapter_size_kb = adapter_path.stat().st_size / 1024

    print(
        f"Saved {len(adapter_weights)} trainable tensors "
        f"({adapter_size_kb:.1f} KB) to {OUTPUT_DIR}"
    )
    print(f"Final validation accuracy: {metrics['final_validation_accuracy']:.4f}")


if __name__ == "__main__":
    main()
