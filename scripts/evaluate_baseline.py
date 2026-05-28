from pathlib import Path

import torch
from torch import nn

from scripts.train_baseline_cnn import train, prepare_data_splits
from src.config.load_config import load_config
from src.detection.detection_metrics import calculate_precision_recall_f1, evaluate_detections
from src.detection.nms import non_max_suppression
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.sliding_window_detection import SlidingWindow


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

config = load_config("../src/config/baseline_config.json")

# Na potrzeby ewaluacji domyslnie uzywamy zapisanego checkpointu.
TRAIN_CNN = False
EVALUATION_SPLIT = "test"

checkpoint_path = Path(config["cnn_training"]["checkpoint_path"])

full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
    data_dir=config["data"]["data_dir"],
    train_ratio=config["data"]["train_ratio"],
    val_ratio=config["data"]["val_ratio"],
    seed=config["seed"]
)

if EVALUATION_SPLIT == "val":
    evaluation_samples = val_samples
elif EVALUATION_SPLIT == "test":
    evaluation_samples = test_samples
else:
    raise ValueError(f"Unknown evaluation split: {EVALUATION_SPLIT}")

print("full images:", len(full_dataset))
print("train images:", len(train_samples))
print("val images:", len(val_samples))
print("test images:", len(test_samples))
print("evaluation split:", EVALUATION_SPLIT)

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

cnn.eval()
detector = SlidingWindow(
    cnn=cnn,
    window_sizes=tuple(config["sliding_window"]["window_sizes"]),
    overlap_ratio=config["sliding_window"]["overlap_ratio"],
    threshold=config["sliding_window"]["threshold"],
    crop_size=config["cnn_dataset"]["crop_size"],
    device=device,
)

total_tp = 0
total_fp = 0
total_fn = 0

for sample in evaluation_samples:
    image_path = sample["image_path"]
    true_bboxes = sample["bboxes"]

    boxes, scores = detector.predict_window(image_path)
    boxes_before_nms = len(boxes)
    boxes, scores = non_max_suppression(
        boxes,
        scores,
        iou_threshold=config["nms"]["iou_threshold"],
    )

    metrics = evaluate_detections(
        pred_boxes=boxes,
        pred_scores=scores,
        true_boxes=true_bboxes,
        iou_threshold=config["metrics"]["iou_threshold"],
    )

    total_tp += metrics["tp"]
    total_fp += metrics["fp"]
    total_fn += metrics["fn"]

    print(
        sample["file_name"],
        "| detections:",
        len(boxes),
        "from:",
        boxes_before_nms,
        "| gt:",
        len(true_bboxes),
        "| TP:",
        metrics["tp"],
        "FP:",
        metrics["fp"],
        "FN:",
        metrics["fn"],
    )

precision, recall, f1 = calculate_precision_recall_f1(
    total_tp,
    total_fp,
    total_fn,
)

print()
print("SUMMARY")
print("split:", EVALUATION_SPLIT)
print("images:", len(evaluation_samples))
print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)
print("precision:", f"{precision:.4f}")
print("recall:", f"{recall:.4f}")
print("f1:", f"{f1:.4f}")
