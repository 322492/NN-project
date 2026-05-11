from pathlib import Path

import torch
from torch import nn

from src.detection.IoU import max_iou_with_true_boxes
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.sliding_window_detection import SlidingWindow
from scripts.train_baseline_cnn import train, prepare_data_splits
from src.detection.bbox_visualization import drew_bbox_and_save
from src.config.load_config import load_config
from src.detection.nms import non_max_suppression

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

config = load_config("../src/config/baseline_config.json")

#TRAIN_CNN = config["cnn_training"]["train_cnn"]
TRAIN_CNN = False
checkpoint_path = Path(config["cnn_training"]["checkpoint_path"])


full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
    data_dir=config["data"]["data_dir"],
    train_ratio=config["data"]["train_ratio"],
    val_ratio=config["data"]["val_ratio"],
    seed=config["seed"]
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
        config=config
    )
else:
    print(f"Loading checkpoint: {checkpoint_path}")
    cnn.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

## sliding window
cnn.eval()
detector = SlidingWindow(
    cnn=cnn,
    window_sizes=tuple(config["sliding_window"]["window_sizes"]),
    overlap_ratio=config["sliding_window"]["overlap_ratio"],
    threshold=config["sliding_window"]["threshold"],
    crop_size=config["cnn_dataset"]["crop_size"],
    device=device,
)

#image_path = test_samples[0]["image_path"]

#konkretne zdj
DEBUG_IMAGE_NAME = "2280.jpg"
# DEBUG_IMAGE_NAME = "1467.jpg"
debug_sample = None

for sample in full_dataset.samples:
    if sample["file_name"] == DEBUG_IMAGE_NAME:
        debug_sample = sample
        break

if debug_sample is None:
    raise ValueError(f"Image not found: {DEBUG_IMAGE_NAME}")

image_path = debug_sample["image_path"]
true_bboxes = debug_sample["bboxes"]

boxes, scores = detector.predict_window(image_path)
boxes_before_nms = len(boxes)
boxes, scores = non_max_suppression(
    boxes,
    scores,
    iou_threshold=config["nms"]["iou_threshold"],
)

print("image:", image_path)
print(
    "number of detections after NMS:",
    len(boxes),
    "from:",
    boxes_before_nms,
    "vs ground truth:",
    len(true_bboxes),
)

print(debug_sample["image_path"])
for box, score in zip(boxes, scores):
    print(box, score, max_iou_with_true_boxes(box, true_bboxes))

drew_bbox_and_save(image_path, boxes, true_bboxes, config)