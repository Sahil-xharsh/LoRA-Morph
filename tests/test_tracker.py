import torch
import torch.nn as nn

from lora_morph import LoRATracker, inject_lora


def test_tracker_records_loss_gradients_and_adapter_rank(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3))
    inject_lora(model, {"0"}, r=2, alpha=4)
    with torch.no_grad():
        model[0].B[0, 0] = 1

    tracker = LoRATracker()
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    tracker.record_loss(loss)
    tracker.record_adapters(model)
    tracker.save(tmp_path / "tracker.json")

    metrics = tracker.to_dict()
    assert metrics["losses"]
    assert metrics["gradient_norms"]["0"]
    assert metrics["singular_values"]["0"]
    assert metrics["effective_ranks"]["0"] == [1]
    assert (tmp_path / "tracker.json").exists()
