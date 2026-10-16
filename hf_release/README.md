---
license: mit
base_model: prajjwal1/bert-tiny
datasets:
- stanfordnlp/sst2
tags:
- lora
- pytorch
- text-classification
---

# LoRA-Morph BERT-tiny SST-2 adapter

LoRA-Morph is a from-scratch PyTorch LoRA implementation. The project is
available at [github.com/Sahil-xharsh/LoRA-Morph](https://github.com/Sahil-xharsh/LoRA-Morph).

## Training setup

This adapter targets the `query` and `value` projections of
`prajjwal1/bert-tiny`. It uses rank 2, alpha 16, seed 42, 2,000 training
examples, 500 validation examples, three epochs, batch size 16, and learning
rate `1e-4`.

## Rank-sweep results

| Rank | Trainable parameters | Final validation accuracy |
| ---: | ---: | ---: |
| 2 | 2,306 | 60.8% |
| 4 | 4,354 | 60.4% |
| 8 | 8,450 | 56.0% |
| 16 | 16,642 | 60.2% |

Rank 2 is published because it had the highest measured accuracy. Differences
between ranks should be interpreted as being within approximately ±2 percentage
points of noise on 500 validation examples.

## PEFT parity

The custom implementation and PEFT had maximum forward deviation `0.0`, adapter
gradient cosine similarity `0.99999998`, equal trainable parameter counts of
`8,450`, and final validation accuracy difference `0.6` percentage points.

## Usage

```python
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

from lora_morph import inject_lora

model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny", num_labels=2
)
inject_lora(model, {"query", "value"}, r=2, alpha=16)
model.load_state_dict(load_file("adapter.safetensors"), strict=False)
model.eval()
```

## Limitations

This is a small-scale demo trained on 2,000 examples with a tiny model. It is
not intended as a general-purpose sentiment classifier.

## License

MIT License.
