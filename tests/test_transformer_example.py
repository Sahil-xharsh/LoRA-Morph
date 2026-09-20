from datasets import Dataset
import pytest

from examples.transformer_example import select_fixed_subset


def test_fixed_subset_is_reproducible():
    dataset = Dataset.from_dict({"value": list(range(20))})

    first = select_fixed_subset(dataset, 5, 42)
    second = select_fixed_subset(dataset, 5, 42)

    assert first["value"] == second["value"]
    assert len(first) == 5


def test_fixed_subset_rejects_bad_size():
    dataset = Dataset.from_dict({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="greater than 0"):
        select_fixed_subset(dataset, 0, 42)

    with pytest.raises(ValueError, match="exceeds dataset size"):
        select_fixed_subset(dataset, 4, 42)

        