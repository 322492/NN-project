from pathlib import Path

import torch
from torch import nn

from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.sliding_window_detection import SlidingWindow
from scripts.train_baseline_cnn import train, prepare_data_splits


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

TRAIN_CNN = False
checkpoint_path = Path("../checkpoints/baseline_cnn.pt")

full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
    data_dir="../data/ena24_sample"
)

print("full images:", len(full_dataset))
print("train images:", len(train_samples))
print("val images:", len(val_samples))
print("test images:", len(test_samples))

cnn = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()

if TRAIN_CNN or not checkpoint_path.exists():
    train(
        cnn=cnn,
        criterion=criterion,
        device=device,
        train_samples=train_samples,
        val_samples=val_samples,
    )
else:
    print(f"Loading checkpoint: {checkpoint_path}")
    cnn.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

cnn.eval()

## sliding window
detector = SlidingWindow(
    cnn=cnn,
    window_sizes=(64, 128, 256),
    overlap_ratio=0.5,
    threshold=0.5,
    crop_size=128,
    device=device,
)

image_path = test_samples[0]["image_path"]

boxes, scores = detector.predict_window(image_path)

print("image:", image_path)
print(
    "number of detections:",
    len(boxes),
    "vs ground truth:",
    len(test_samples[0]["bboxes"]),
)

for box, score in zip(boxes[:10], scores[:10]):
    print(box, score)