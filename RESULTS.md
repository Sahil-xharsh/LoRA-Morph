# LoRA-Morph Results

All recorded runs use seed `42` and CPU execution unless noted otherwise.
Metrics are stored as JSON under `results/assets/metrics/`.

## Transformer baseline

Model: `prajjwal1/bert-tiny`, SST-2, 2,000 training examples, 500 validation
examples, three epochs, batch size 16, learning rate `2e-5`.

- Final validation loss: `0.6356`
- Final validation accuracy: `67.4%`

## Custom transformer LoRA

The custom implementation targets `query` and `value`, using rank 8 and alpha
16. The classifier head remains trainable.

- Trainable parameters: `8,450 / 4,394,370`
- Trainable fraction: `0.192%`
- Final validation loss: `0.6856`
- Final validation accuracy: `53.2%`

## CNN validation

The Fashion-MNIST CNN targets `fc1` and `fc2`, using rank 4 and alpha 8. The
convolutional layers remain unchanged.

- Dataset subset: 5,000 training examples and 1,000 test examples
- Trainable parameters: `6,824 / 112,690`
- Trainable fraction: `6.06%`
- Final test loss: `1.2345`
- Final test accuracy: `53.1%`

## Rank sweep

The fixed SST-2 setup was run with alpha 16 for each rank:

| Rank | Trainable parameters | Final validation accuracy |
| ---: | ---: | ---: |
| 2 | 2,306 | 53.8% |
| 4 | 4,354 | 50.8% |
| 8 | 8,450 | 52.2% |
| 16 | 16,642 | 52.6% |

## PEFT comparison verdicts

Custom LoRA and PEFT used the same model, target modules, rank, alpha, seed,
dataset, and optimizer settings for the comparison run.

| Check | Measured value | Verdict |
| --- | ---: | --- |
| Maximum forward deviation | `0.0` | PASS, threshold `<= 1e-5` |
| Adapter-gradient cosine similarity | `0.99999998` | PASS, threshold `>= 0.999` |
| Trainable parameter count | `8,450` vs `8,450` | PASS |
| Validation curve RMSE | `0.00054` | Reported |
| Final validation accuracy difference | `0.6 percentage points` | PASS, threshold `<= 5 points` |
| Custom loss from initial to final | `0.6987` to `0.6791` | PASS |
| PEFT loss from initial to final | `0.6987` to `0.6801` | PASS |

The full comparison uses learning rate `1e-4` because the original `2e-5`
setting did not reduce the validation loss over the short three-epoch run.
The adjusted value is recorded in `peft_comparison.json` and is shared by
both implementations.

## Release checks

- Unit and integration tests: `18 passed`.
- Adapter export: generated `hf_release/adapter.pt`, `config.json`, and
  `metrics.json`.
- Package metadata: version `0.1.0`, Python `>=3.11`.
- Remaining owner decision: choose and add a project license and contribution
  policy before publishing the repository.
