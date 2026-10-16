import json

import torch
from safetensors.torch import load_file

from lora_morph import inject_lora
from scripts.export_hf import export_bundle


def test_export_bundle_writes_only_adapter_and_head_tensors(tmp_path, monkeypatch):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(4, 4)
            self.classifier = torch.nn.Linear(4, 2)

        def forward(self, inputs):
            return self.classifier(self.query(inputs))

    model = ToyModel()
    inject_lora(model, {"query"}, r=2, alpha=4)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": state}, checkpoint)
    metrics = tmp_path / "rank_sweep.json"
    metrics.write_text(
        json.dumps(
            {"results": [{"rank": 2, "history": [{"validation_accuracy": 0.6}]}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.export_hf.METRICS_PATH", metrics)
    monkeypatch.setattr("scripts.export_hf.version", lambda _: "0.1.0")
    monkeypatch.setattr("scripts.export_hf.verify_roundtrip", lambda *args: 0.0)

    output = tmp_path / "hf_release"
    config = export_bundle(checkpoint, 2, 4, output)
    exported = load_file(str(output / "adapter.safetensors"))

    assert set(exported) == set(state)
    assert all(
        ".A" in name or ".B" in name or name.startswith("classifier.")
        for name in exported
    )
    assert config["base_model"] == "prajjwal1/bert-tiny"
    assert config["r"] == 2
    assert config["alpha"] == 4
    assert config["final_val_accuracy"] == 0.6
    assert json.loads((output / "config.json").read_text())["task"] == "sst2-sentiment"
