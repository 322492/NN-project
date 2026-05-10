from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.datasets.ena24_dataset import ENA24DetectionDataset
from src.datasets.ena24_window_dataset import ENA24WindowDataset
from src.models.baseline.cnn_classifier import SimpleCNN

torch.manual_seed(42)

full_dataset = ENA24DetectionDataset(
    data_dir="../data/ena24_sample",
)

train_dataset_size = int(0.8 * len(full_dataset))
train_size = int(0.9 * train_dataset_size)
val_size = int(train_dataset_size - train_size)
test_size = len(full_dataset) - train_size - val_size

train_detection_dataset, val_detection_dataset, test_detection_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

## zdj mają różne rozmiary
# train_detection_loader = DataLoader(train_detection_dataset, batch_size=1, shuffle=True)
# val_detection_loader = DataLoader(val_detection_dataset, batch_size=1, shuffle=False)
# test_detection_loader = DataLoader(test_detection_dataset, batch_size=1, shuffle=False)

## dataset dla CNN
train_samples = [full_dataset.samples[i] for i in train_detection_dataset.indices]
val_samples = [full_dataset.samples[i] for i in val_detection_dataset.indices]
test_samples = [full_dataset.samples[i] for i in test_detection_dataset.indices]

train_window_dataset = ENA24WindowDataset(train_samples)
val_window_dataset = ENA24WindowDataset(val_samples)

train_window_loader = DataLoader(train_window_dataset, batch_size=32, shuffle=True)
val_window_loader = DataLoader(val_window_dataset, batch_size=32, shuffle=False)

print("full images:", len(full_dataset))
print("train images:", len(train_samples))
print("val images:", len(val_samples))
print("test images:", len(test_samples))
print("train CNN crops:", len(train_window_dataset))
print("val CNN crops:", len(val_window_dataset))


## train CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 10

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model,
        train_window_loader,
        optimizer,
        criterion,
        device,
    )

    val_loss, val_acc = evaluate(
        model,
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
torch.save(model.state_dict(), "../checkpoints/baseline_cnn.pt")

##