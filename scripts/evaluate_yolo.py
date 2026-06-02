#!/usr/bin/env python3
"""
Evaluate a trained YOLO checkpoint on the same ENA24 split as the baseline.

Uses data.coco_dir + seed + ratios + size from yolo config (identical to
prepare_data_splits / evaluate_baseline). Metrics are class-agnostic (IoU only).

Default: src/config/yolo_config.json, test split.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
import click
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.config.paths import normalize_config_paths, resolve_project_path
from src.datasets.data_splits import prepare_data_splits
from src.detection.detection_metrics import (
    calculate_map,
    calculate_precision_recall_f1,
    evaluate_detections,
)
from src.detection.IoU import match_true_boxes_with_predictions, mean_iou
from src.utils.wandb_utils import (
    wandb_log,
    finish_wandb,
    init_wandb,
    log_model_artifact,
)

def resolve_inference_device(device_setting: str):
    if device_setting != "auto":
        return device_setting

    if torch.cuda.is_available():
        return 0

    return "cpu"


def yolo_results_to_boxes_and_scores(result) -> tuple[list[list[float]], list[float]]:
    boxes = []
    scores = []

    if result.boxes is None or len(result.boxes) == 0:
        return boxes, scores

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(float(box.conf[0]))

    return boxes, scores


def run_evaluation(
    config: dict,
    evaluation_split: str,
    checkpoint_path: Path | None = None,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from error

    seed = config["seed"]
    torch.manual_seed(seed)

    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})
    metrics_cfg = config.get("metrics", {})
    checkpoints_cfg = config["checkpoints"]

    if checkpoint_path is None:
        checkpoint_path = Path(checkpoints_cfg["best_checkpoint_path"])

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_inference_device(config.get("training", {}).get("device", "auto"))
    conf_threshold = inference_cfg.get("conf_threshold", 0.25)
    nms_iou_threshold = inference_cfg.get("iou_threshold", 0.45)
    match_iou_threshold = metrics_cfg.get("iou_threshold", 0.5)

    full_dataset, train_samples, val_samples, test_samples = prepare_data_splits(
        data_dir=data_cfg["coco_dir"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=seed,
        size=data_cfg.get("size"),
    )

    if evaluation_split == "val":
        evaluation_samples = val_samples
    elif evaluation_split == "test":
        evaluation_samples = test_samples
    else:
        raise ValueError(f"Unknown evaluation split: {evaluation_split}")

    print("device:", device)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("COCO dir (split source):", data_cfg["coco_dir"])
    print("full dataset images:", len(full_dataset))
    print("used images:", len(train_samples) + len(val_samples) + len(test_samples))
    print("train images:", len(train_samples))
    print("val images:", len(val_samples))
    print("test images:", len(test_samples))
    print("evaluation split:", evaluation_split)
    print("conf_threshold:", conf_threshold)
    print("nms_iou_threshold:", nms_iou_threshold)
    print("match_iou_threshold:", match_iou_threshold)

    model = YOLO(str(checkpoint_path))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_boxes = 0
    sum_iou = 0.0
    total_detections = 0
    all_pred_boxes = []
    all_pred_scores = []
    all_true_boxes = []

    for idx, sample in enumerate(evaluation_samples):
        image_path = sample["image_path"]
        true_bboxes = sample["bboxes"]

        predict_results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=nms_iou_threshold,
            device=device,
            verbose=False,
        )

        boxes, scores = yolo_results_to_boxes_and_scores(predict_results[0])

        metrics = evaluate_detections(
            pred_boxes=boxes,
            pred_scores=scores,
            true_boxes=true_bboxes,
            iou_threshold=match_iou_threshold,
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_detections += len(boxes)
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

    avg_detections = total_detections / n_images if n_images > 0 else 0.0
    mean_iou_value = mean_iou(sum_iou, total_true_boxes)
    map_score = calculate_map(
        all_pred_boxes,
        all_pred_scores,
        all_true_boxes,
        iou_threshold=match_iou_threshold,
    )

    summary = {
        "model": "yolo",
        "checkpoint": str(checkpoint_path.resolve()),
        "config_description": config.get("description", ""),
        "coco_dir": data_cfg["coco_dir"],
        "seed": seed,
        "data_size": data_cfg.get("size"),
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
        "avg_detections": avg_detections,
        "conf_threshold": conf_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "match_iou_threshold": match_iou_threshold,
    }

    print()
    print("SUMMARY")
    print("model:", summary["model"])
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
    print("avg detections per image:", f"{summary['avg_detections']:.4f}")

    return summary


def save_summary_json(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print("Saved results:", output_path)


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "yolo_config.json"),
    show_default=True,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Override checkpoints.best_checkpoint_path from config.",
)
@click.option(
    "--split",
    "evaluation_split",
    type=click.Choice(["test", "val"], case_sensitive=False),
    default="test",
    show_default=True,
)
@click.option(
    "--output_json",
    type=click.Path(dir_okay=False),
    default=None,
    help="Save SUMMARY to JSON, e.g. results/yolo_sample_test.json",
)
def main(
    config_path: str,
    checkpoint: str | None,
    evaluation_split: str,
    output_json: str | None,
):
    config = normalize_config_paths(load_config(config_path))
    print("Config:", config_path)

    checkpoint_path = Path(checkpoint) if checkpoint else None
    summary = run_evaluation(
        config=config,
        evaluation_split=evaluation_split,
        checkpoint_path=checkpoint_path,
    )

    wandb_cfg = config.get("wandb", {})

    run = init_wandb(
        config=config,
        enabled=wandb_cfg.get("enabled", False),
        project=wandb_cfg.get("project", "ena24-yolo"),
        job_type="eval",
    )

    wandb_log(run, summary)
    finish_wandb(run)

    if output_json is not None:
        save_summary_json(summary, resolve_project_path(output_json))


if __name__ == "__main__":
    main()
