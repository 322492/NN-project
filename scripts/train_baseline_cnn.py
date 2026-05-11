from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.datasets.ena24_dataset import ENA24DetectionDataset
from src.datasets.ena24_window_dataset import ENA24WindowDataset
from src.models.baseline.cnn_classifier import train_one_epoch, evaluate


def prepare_data_splits(data_dir="../data/ena24_sample"):
    torch.manual_seed(42)

    full_dataset = ENA24DetectionDataset(
        data_dir=data_dir,
    )

    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_detection_dataset, val_detection_dataset, test_detection_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_samples = [full_dataset.samples[i] for i in train_detection_dataset.indices]
    val_samples = [full_dataset.samples[i] for i in val_detection_dataset.indices]
    test_samples = [full_dataset.samples[i] for i in test_detection_dataset.indices]

    return full_dataset, train_samples, val_samples, test_samples


def train(cnn, criterion, device, train_samples, val_samples):
    train_window_dataset = ENA24WindowDataset(train_samples)
    val_window_dataset = ENA24WindowDataset(val_samples)

    train_window_loader = DataLoader(train_window_dataset, batch_size=32, shuffle=True)
    val_window_loader = DataLoader(val_window_dataset, batch_size=32, shuffle=False)

    print("train images:", len(train_samples))
    print("val images:", len(val_samples))
    print("train CNN crops:", len(train_window_dataset))
    print("val CNN crops:", len(val_window_dataset))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            cnn,
            train_window_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = evaluate(
            cnn,
            val_window_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

    Path("../checkpoints").mkdir(exist_ok=True)
    torch.save(cnn.state_dict(), "../checkpoints/baseline_cnn.pt")
    print("Saved checkpoint: ../checkpoints/baseline_cnn.pt")