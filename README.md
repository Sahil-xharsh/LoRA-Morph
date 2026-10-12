# LoRA-Morph

LoRA-Morph is a small, framework-agnostic LoRA implementation built from
scratch in PyTorch. The project checks the same adapter implementation against
three settings: a standalone linear layer, a BERT sentiment classifier, and a
CNN image classifier.

## What is included

- `LoRALinear`, with zero-initialized `B` for an exact initial no-op.
- `inject_lora()`, which matches exact leaf module names and freezes the base
  model.
- SST-2 experiments using `prajjwal1/bert-tiny`.
- Fashion-MNIST CNN validation.
- Adapter tracking for loss, gradient norms, singular values, and effective
  ranks.
- Rank sweep and numerical comparison with Hugging Face PEFT.

## Install

Python 3.11 or newer is required. From the repository root:

```bash
uv sync
```

The project uses PyTorch, Transformers, Datasets, Torchvision, and PEFT. The
first experiment run downloads the public model and dataset files.

## Tests and release checks

```bash
uv run pytest -q --basetemp .pytest-tmp
uv run ruff check .
uv run ruff format --check .
uv build
```

The workspace-local pytest temp directory is useful on systems where the
default temporary directory is restricted.

## Experiments

Run the verified experiments with:

```bash
uv run python -m examples.transformer_example
uv run python -m examples.lora_transformer_example
uv run python -m examples.cnn_example
uv run python -m examples.rank_sweep
uv run python -m examples.peft_comparison
```

The transformer runs use a seed-42 shuffle of 2,000 SST-2 training examples
and 500 validation examples. The baseline uses full fine-tuning. The LoRA run
targets BERT's `query` and `value` projections with rank 8 and alpha 16.

The CNN run uses 5,000 Fashion-MNIST training examples and 1,000 test
examples. It targets the `fc1` and `fc2` linear layers with rank 4 and alpha 8.

## Adapter export

The export helper writes trainable tensors and metadata for the BERT LoRA run:

```bash
uv run python -m examples.export_adapter
```

The generated bundle contains `adapter.pt`, `config.json`, and `metrics.json`
under `hf_release/`. The adapter requires the base model named in
`hf_release/config.json` for loading.

## Artifacts

- Metrics: `results/assets/metrics/`
- Export bundle: `hf_release/`
- Build output: `dist/`

See [`RESULTS.md`](RESULTS.md) for the recorded experiment values and release
verdicts.

## Runtime and hardware

The transformer experiments take approximately 10 to 20 minutes on CPU and
roughly 2 to 5 minutes on a typical CUDA GPU, depending on batch size and
environment. The CNN and parity checks are smaller, but the rank sweep runs
the transformer training four times.

