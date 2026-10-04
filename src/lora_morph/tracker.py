import json
from pathlib import Path

import torch
import torch.nn as nn

from lora_morph.lora import LoRALinear


class LoRATracker:
    def __init__(self):
        self.losses = []
        self.gradient_norms = {}
        self.singular_values = {}
        self.effective_ranks = {}

    def record_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().item()
        self.losses.append(float(loss))

    def record_adapters(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue

            if module.A.grad is None or module.B.grad is None:
                continue

            update = module.A @ module.B
            singular_values = torch.linalg.svdvals(update.detach()).cpu().tolist()
            threshold = singular_values[0] * 1e-6 if singular_values else 0
            effective_rank = sum(value > threshold for value in singular_values)
            gradient_norm = torch.cat(
                [module.A.grad.detach().flatten(), module.B.grad.detach().flatten()]
            ).norm()

            self.gradient_norms.setdefault(name, []).append(float(gradient_norm))
            self.singular_values.setdefault(name, []).append(singular_values)
            self.effective_ranks.setdefault(name, []).append(effective_rank)

    def to_dict(self):
        return {
            "losses": self.losses,
            "gradient_norms": self.gradient_norms,
            "singular_values": self.singular_values,
            "effective_ranks": self.effective_ranks,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
