import sys
from pathlib import Path

import click
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.data_splits import prepare_data_splits_from_data_config
from src.config.load_config import load_config
from src.config.paths import normalize_config_paths
from src.detection.detection_metrics import (
    calculate_map,
    calculate_precision_recall_f1,
    evaluate_detections,
)
from src.detection.IoU import match_true_boxes_with_predictions, mean_iou
from src.detection.nms import non_max_suppression
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.resnet_classifier import ResNetBinaryClassifier
from src.models.baseline.sliding_window_detection import SlidingWindow


def build_cnn(model_name: str, device: torch.device):
    if model_name == "resnet":
        return ResNetBinaryClassifier(pretrained=True).to(device)
    if model_name == "simple_cnn":
        return SimpleCNN().to(device)
    raise ValueError(f"Unknown model: {model_name}")


def run_evaluation(config: dict, evaluation_split: str) -> dict:
    seed = config["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint_path = Path(config["cnn_training"]["best_checkpoint_path"])

    full_dataset, train_samples, val_samples, test_samples = prepare_data_splits_from_data_config(
        config["data"],
        data_dir=config["data"]["data_dir"],
        seed=seed,
    )

    if evaluation_split == "val":
        evaluation_samples = val_samples
    elif evaluation_split == "test":
        evaluation_samples = test_samples
    else:
        raise ValueError(f"Unknown evaluation split: {evaluation_split}")

    print("full dataset images:", len(full_dataset))
    print("used images:", len(train_samples) + len(val_samples) + len(test_samples))
    print("train images:", len(train_samples))
    print("val images:", len(val_samples))
    print("test images:", len(test_samples))
    print("evaluation split:", evaluation_split)

    model_name = config["cnn_training"].get("model", "simple_cnn")
    cnn = build_cnn(model_name, device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn.load_state_dict(checkpoint)
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
    all_pred_boxes = []
    all_pred_scores = []
    all_true_boxes = []

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
        all_pred_boxes.append(boxes)
        all_pred_scores.append(scores)
        all_true_boxes.append(true_bboxes)

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

    precision, recall, f1 = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    n_images = len(evaluation_samples)

    avg_boxes_before_nms = total_boxes_before_nms / n_images if n_images > 0 else 0.0
    avg_boxes_after_nms = total_boxes_after_nms / n_images if n_images > 0 else 0.0
    mean_iou_value = mean_iou(sum_iou, total_true_boxes)
    map_score = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_true_boxes,
        iou_threshold=config["metrics"]["iou_threshold"],
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": evaluation_split,
        "images": n_images,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map": map_score,
        "mean_iou": mean_iou_value,
        "avg_boxes_before_nms": avg_boxes_before_nms,
        "avg_boxes_after_nms": avg_boxes_after_nms,
    }

    print()
    print("SUMMARY")
    print("checkpoint:", summary["checkpoint"])
    print("split:", summary["split"])
    print("images:", summary["images"])
    print("TP:", summary["tp"])
    print("FP:", summary["fp"])
    print("FN:", summary["fn"])
    print("precision:", f"{summary['precision']:.4f}")
    print("recall:", f"{summary['recall']:.4f}")
    print("f1:", f"{summary['f1']:.4f}")
    print("mAP:", f"{summary['map']:.4f}")
    print("mean_iou:", f"{summary['mean_iou']:.4f}")
    print("avg boxes before NMS:", f"{summary['avg_boxes_before_nms']:.4f}")
    print("avg boxes after NMS:", f"{summary['avg_boxes_after_nms']:.4f}")

    return summary


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "baseline_config.json"),
    show_default=True,
)
@click.option(
    "--split",
    "evaluation_split",
    type=click.Choice(["test", "val"], case_sensitive=False),
    default="test",
    show_default=True,
)
def main(config_path: str, evaluation_split: str):
    config = load_config(config_path)
    config = normalize_config_paths(config)
    print("Config:", config_path)
    run_evaluation(config, evaluation_split)


if __name__ == "__main__":
    main()
