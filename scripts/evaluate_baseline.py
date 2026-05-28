from pathlib import Path

import torch
from torch import nn

from scripts.train_baseline_cnn import train, prepare_data_splits
from src.config.load_config import load_config
from src.detection.detection_metrics import calculate_precision_recall_f1, evaluate_detections
from src.detection.nms import non_max_suppression
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.sliding_window_detection import SlidingWindow
from src.models.baseline.resnet_classifier import ResNetBinaryClassifier
from src.detection.IoU import max_iou_with_true_boxes, match_true_boxes_with_predictions, mean_iou


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

config = load_config("../src/config/baseline_config.json")

# Na potrzeby ewaluacji domyslnie uzywamy zapisanego checkpointu.

EVALUATION_SPLIT = "test"
# EVALUATION_SPLIT = "val"

checkpoint_path = Path(config["cnn_training"]["best_checkpoint_path"])


full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
    data_dir=config["data"]["data_dir"],
    train_ratio=config["data"]["train_ratio"],
    val_ratio=config["data"]["val_ratio"],
    seed=config["seed"],
    size=config["data"].get("size"),
)

if EVALUATION_SPLIT == "val":
    evaluation_samples = val_samples
elif EVALUATION_SPLIT == "test":
    evaluation_samples = test_samples
else:
    raise ValueError(f"Unknown evaluation split: {EVALUATION_SPLIT}")

print("full dataset images:", len(full_dataset))
print("used images:", len(train_samples) + len(val_samples) + len(test_samples))
print("train images:", len(train_samples))
print("val images:", len(val_samples))
print("test images:", len(test_samples))
print("evaluation split:", EVALUATION_SPLIT)

model_name = config["cnn_training"].get("model", "simple_cnn")

if model_name == "resnet":
    cnn = ResNetBinaryClassifier(pretrained=True).to(device)
elif model_name == "simple_cnn":
    cnn = SimpleCNN().to(device)
else:
    raise ValueError(f"Unknown model: {model_name}")

criterion = nn.BCEWithLogitsLoss()

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

total_true_boxes = 0
sum_iou = 0.0

total_boxes_before_nms = 0
total_boxes_after_nms = 0

for idx, sample in enumerate(evaluation_samples):
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

    total_boxes_before_nms += boxes_before_nms
    total_boxes_after_nms += len(boxes)

    matches = match_true_boxes_with_predictions(
        true_bboxes=true_bboxes,
        pred_boxes=boxes,
        pred_confidence_scores=scores,
    )

    for match in matches:
        total_true_boxes += 1
        sum_iou += match["iou"]

    print(
        f"{idx + 1}/{len(evaluation_samples)}",
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

avg_boxes_before_nms = (
    total_boxes_before_nms / len(evaluation_samples)
    if len(evaluation_samples) > 0
    else 0.0
)

avg_boxes_after_nms = (
    total_boxes_after_nms / len(evaluation_samples)
    if len(evaluation_samples) > 0
    else 0.0
)

mean_iou_value = mean_iou(sum_iou, total_true_boxes)

print()
print("SUMMARY")
print("checkpoint:", checkpoint_path)
print("split:", EVALUATION_SPLIT)
print("images:", len(evaluation_samples))
print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)
print("precision:", f"{precision:.4f}")
print("recall:", f"{recall:.4f}")
print("f1:", f"{f1:.4f}")
print("mean_iou:", f"{mean_iou_value:.4f}")
print("avg boxes before NMS:", f"{avg_boxes_before_nms:.4f}")
print("avg boxes after NMS:", f"{avg_boxes_after_nms:.4f}")