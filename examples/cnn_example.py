import json
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lora_morph import inject_lora
from examples.transformer_example import SEED, set_seed


TRAIN_SIZE = 5000
TEST_SIZE = 1000
BATCH_SIZE = 64
EPOCHS = 2
LR = 1e-3

LORA_RANK = 4
LORA_ALPHA = 8

DATA_DIR = Path("data/fashion_mnist")
RESULT_FILE = Path("results/assets/metrics/cnn_lora.json")


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def add_lora(model):
    return inject_lora(
        model,
        {"fc1", "fc2"},
        r=LORA_RANK,
        alpha=LORA_ALPHA
    )


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_data():
    transform = transforms.ToTensor()

    train = datasets.FashionMNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test = datasets.FashionMNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    g = torch.Generator().manual_seed(SEED)

    train_idx = torch.randperm(len(train), generator=g)[:TRAIN_SIZE]
    test_idx = torch.randperm(len(test), generator=g)[:TEST_SIZE]

    return Subset(train, train_idx), Subset(test, test_idx)


def evaluate(model, loader, device):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    loss_total = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            loss_total += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

    return loss_total / count, correct / count


def train():
    set_seed(SEED)

    train_data, test_data = load_data()

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)

    model = FashionCNN()
    model = add_lora(model)
    model.to(device)

    trainable, total = count_parameters(model)
    print(f"trainable parameters: {trainable}/{total}")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR
    )

    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(EPOCHS):
        model.train()

        loss_total = 0.0
        count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            loss_total += loss.item() * labels.size(0)
            count += labels.size(0)

        train_loss = loss_total / count
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"epoch {epoch + 1}: "
            f"train={train_loss:.4f}, "
            f"test={test_loss:.4f}, "
            f"acc={test_accuracy:.4f}"
        )

    return {
        "experiment": "cnn_lora",
        "dataset": "Fashion-MNIST",
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "device": str(device),
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": ["fc1", "fc2"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "package_versions": {
            "torch": version("torch"),
            "torchvision": version("torchvision")
        },
        "history": history
    }


def main():
    results = train()

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
    )

    print("saved:", RESULT_FILE)


if __name__ == "__main__":
    main()