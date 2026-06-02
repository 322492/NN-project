#!/usr/bin/env python3
"""
Train Ultralytics YOLO on a prepared ENA24 YOLO dataset.

Default:
    src/config/yolo_config.json

Full run:
    --config_path src/config/yolo_config_full.json

Recommended:
    python scripts/train_yolo.py \
        --config_path src/config/yolo_config.json \
        --prepare \
        --epochs 50 \
        --eval_after_train \
        --eval_split val \
        --eval_output_json results/yolo_val.json
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_yolo import run_evaluation, save_summary_json
from scripts.prepare_yolo_dataset import prepare_yolo_dataset
from src.config.load_config import load_config
from src.config.paths import (
    normalize_config_paths,
    resolve_data_yaml_path,
    resolve_project_path,
)
from src.utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_model_artifact,
)


def resolve_train_device(device_setting: str):
    if device_setting != "auto":
        return device_setting

    import torch

    if torch.cuda.is_available():
        return 0

    return "cpu"


def export_best_weights(run_weights_dir: Path, destination_path: Path) -> Path:
    best_weights_path = run_weights_dir / "best.pt"

    if not best_weights_path.exists():
        raise FileNotFoundError(f"YOLO best weights not found: {best_weights_path}")

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_path, destination_path)

    print(f"Exported checkpoint: {destination_path}")
    return destination_path


def to_float(value) -> float:
    """
    Twarda konwersja na Python float.
    W&B wtedy powinien widzieć wartość jako metrykę numeryczną.
    """
    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def to_int(value) -> int:
    """
    Twarda konwersja na Python int.
    """
    if hasattr(value, "item"):
        value = value.item()

    return int(value)


def wandb_log_numeric(run, values: dict) -> None:
    """
    Bezpośrednie run.log zamiast wrappera, żeby uniknąć przypadkowej
    konwersji wartości na string.
    """
    if run is None:
        return

    run.log(values)


def wandb_summary_update(run, values: dict) -> None:
    """
    W&B table / porównywanie runów bierze finalne kolumny głównie z summary.
    """
    if run is None:
        return

    if not hasattr(run, "summary"):
        return

    for key, value in values.items():
        run.summary[key] = value


def log_train_summary_to_wandb(
    run,
    epochs: int,
    training_cfg: dict,
    config: dict,
    run_save_dir: Path,
    exported_checkpoint: Path,
) -> None:
    train_summary = {
        "train_epochs_num": to_int(epochs),
        "train_imgsz_num": to_int(training_cfg.get("imgsz", 640)),
        "train_batch_num": to_int(training_cfg.get("batch", 16)),
        "train_workers_num": to_int(training_cfg.get("workers", 4)),
        "train_patience_num": to_int(training_cfg.get("patience", 10)),
        "train_seed_num": to_int(config.get("seed", 42)),
    }

    train_info = {
        "train_run_save_dir": str(run_save_dir),
        "train_best_checkpoint_path": str(exported_checkpoint),
    }

    wandb_log_numeric(run, train_summary)
    wandb_summary_update(run, train_summary)
    wandb_summary_update(run, train_info)


def log_eval_summary_to_wandb(run, eval_split: str, eval_summary: dict) -> None:
    """
    Loguje evaluation jako jawne wartości numeryczne.

    Nazwy mają suffix _num, żeby W&B nie używał starych typów kolumn,
    np. gdy wcześniej eval_val_f1 było zapisane jako tekst.
    """
    if eval_summary is None:
        return

    prefix = f"eval_{eval_split.lower()}"

    eval_metrics = {
        f"{prefix}_precision_num": to_float(eval_summary["precision"]),
        f"{prefix}_recall_num": to_float(eval_summary["recall"]),
        f"{prefix}_f1_num": to_float(eval_summary["f1"]),
        f"{prefix}_map_num": to_float(eval_summary["map"]),
        f"{prefix}_mean_iou_num": to_float(eval_summary["mean_iou"]),
        f"{prefix}_avg_detections_num": to_float(eval_summary["avg_detections"]),
        f"{prefix}_conf_threshold_num": to_float(eval_summary["conf_threshold"]),
        f"{prefix}_nms_iou_threshold_num": to_float(eval_summary["nms_iou_threshold"]),
        f"{prefix}_match_iou_threshold_num": to_float(eval_summary["match_iou_threshold"]),
        f"{prefix}_tp_num": to_int(eval_summary["tp"]),
        f"{prefix}_fp_num": to_int(eval_summary["fp"]),
        f"{prefix}_fn_num": to_int(eval_summary["fn"]),
        f"{prefix}_images_num": to_int(eval_summary["images"]),
    }

    wandb_log_numeric(run, eval_metrics)
    wandb_summary_update(run, eval_metrics)


def ensure_dataset_ready(
    config: dict,
    prepare_if_missing: bool,
    copy_images: bool,
) -> Path:
    data_yaml_path = resolve_data_yaml_path(config)

    if data_yaml_path.exists():
        return data_yaml_path

    if not prepare_if_missing:
        raise FileNotFoundError(
            f"YOLO data.yaml not found: {data_yaml_path}\n"
            "Run prepare_yolo_dataset.py first, or pass --prepare."
        )

    data_cfg = config["data"]
    detection_cfg = config.get("detection", {})

    print("Preparing YOLO dataset because data.yaml is missing...")

    prepare_yolo_dataset(
        coco_dir=resolve_project_path(data_cfg["coco_dir"]),
        output_dir=resolve_project_path(data_cfg["output_dir"]),
        seed=config["seed"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        size=data_cfg.get("size"),
        copy_images=copy_images or bool(data_cfg.get("copy_images", False)),
        single_class=bool(detection_cfg.get("single_class", True)),
        class_name=detection_cfg.get("class_name", "object"),
    )

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Failed to create data.yaml: {data_yaml_path}")

    return data_yaml_path


def train_yolo(
    config: dict,
    epochs_override: int | None = None,
    resume_path: str | None = None,
    prepare_if_missing: bool = False,
    copy_images: bool = False,
    eval_after_train: bool = False,
    eval_split: str = "test",
    eval_output_json: str | None = None,
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required for YOLO training. "
            "Install with: pip install ultralytics"
        ) from error

    data_yaml_path = ensure_dataset_ready(
        config=config,
        prepare_if_missing=prepare_if_missing,
        copy_images=copy_images,
    )

    model_cfg = config["model"]
    training_cfg = config["training"]
    checkpoints_cfg = config["checkpoints"]
    wandb_cfg = config.get("wandb", {})

    weights = model_cfg.get("weights", "yolov8n.pt")
    epochs = epochs_override or training_cfg["epochs"]
    device = resolve_train_device(training_cfg.get("device", "auto"))

    print("Config description:", config.get("description", ""))
    print("data.yaml:", data_yaml_path)
    print("weights:", weights)
    print("epochs:", epochs)
    print("device:", device)
    print("project:", training_cfg["project"])
    print("name:", training_cfg["name"])
    print("eval_after_train:", eval_after_train)
    print("eval_split:", eval_split)

    if resume_path:
        model = YOLO(resume_path)
    else:
        model = YOLO(weights)

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "imgsz": training_cfg.get("imgsz", 640),
        "batch": training_cfg.get("batch", 16),
        "device": device,
        "project": training_cfg["project"],
        "name": training_cfg["name"],
        "patience": training_cfg.get("patience", 10),
        "workers": training_cfg.get("workers", 4),
        "seed": config.get("seed", 42),
        "pretrained": training_cfg.get("pretrained", True),
        "exist_ok": True,
    }

    if resume_path:
        train_kwargs["resume"] = True

    run = init_wandb(
        config={
            "config": config,
            "train_kwargs": train_kwargs,
            "weights": weights,
            "epochs": int(epochs),
            "device": device,
            "eval_after_train": bool(eval_after_train),
            "eval_split": eval_split,
        },
        enabled=wandb_cfg.get("enabled", False),
        project=wandb_cfg.get("project", "ena24-yolo"),
        job_type="train",
    )

    try:
        results = model.train(**train_kwargs)

        run_save_dir = (
            Path(results.save_dir)
            if hasattr(results, "save_dir")
            else Path(model.trainer.save_dir)
        )
        weights_dir = run_save_dir / "weights"

        exported_checkpoint = export_best_weights(
            run_weights_dir=weights_dir,
            destination_path=checkpoints_cfg["best_checkpoint_path"],
        )

        log_train_summary_to_wandb(
            run=run,
            epochs=epochs,
            training_cfg=training_cfg,
            config=config,
            run_save_dir=run_save_dir,
            exported_checkpoint=exported_checkpoint,
        )

        if run is not None:
            log_model_artifact(
                run=run,
                checkpoint_path=exported_checkpoint,
                artifact_name="yolo_best",
            )

        if eval_after_train:
            print()
            print(f"Running YOLO evaluation after training on split: {eval_split}")

            eval_summary = run_evaluation(
                config=config,
                evaluation_split=eval_split,
                checkpoint_path=exported_checkpoint,
            )

            log_eval_summary_to_wandb(
                run=run,
                eval_split=eval_split,
                eval_summary=eval_summary,
            )

            if eval_output_json is not None:
                save_summary_json(
                    eval_summary,
                    resolve_project_path(eval_output_json),
                )

        print("Training run directory:", run_save_dir)
        print("Exported best checkpoint:", exported_checkpoint)

        return results

    finally:
        finish_wandb(run)


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "yolo_config.json"),
    show_default=True,
    help="Sample: yolo_config.json | Full: yolo_config_full.json",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override training.epochs from config.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Resume from a YOLO checkpoint .pt, e.g. runs/detect/.../weights/last.pt",
)
@click.option(
    "--prepare",
    "prepare_dataset",
    is_flag=True,
    default=False,
    help="Run dataset preparation before training.",
)
@click.option(
    "--copy_images",
    is_flag=True,
    default=False,
    help="Copy images during --prepare instead of using symlinks.",
)
@click.option(
    "--eval_after_train",
    is_flag=True,
    default=False,
    help="Run evaluation after YOLO training and log metrics to the same W&B run.",
)
@click.option(
    "--eval_split",
    type=click.Choice(["test", "val"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Dataset split used for post-training evaluation.",
)
@click.option(
    "--eval_output_json",
    type=click.Path(dir_okay=False),
    default=None,
    help="Save evaluation summary JSON, e.g. results/yolo_test.json.",
)
def main(
    config_path: str,
    epochs: int | None,
    resume: str | None,
    prepare_dataset: bool,
    copy_images: bool,
    eval_after_train: bool,
    eval_split: str,
    eval_output_json: str | None,
):
    config = normalize_config_paths(load_config(config_path))

    if prepare_dataset:
        data_cfg = config["data"]
        detection_cfg = config.get("detection", {})

        print("Preparing YOLO dataset...")

        prepare_yolo_dataset(
            coco_dir=resolve_project_path(data_cfg["coco_dir"]),
            output_dir=resolve_project_path(data_cfg["output_dir"]),
            seed=config["seed"],
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            size=data_cfg.get("size"),
            copy_images=copy_images or bool(data_cfg.get("copy_images", False)),
            single_class=bool(detection_cfg.get("single_class", True)),
            class_name=detection_cfg.get("class_name", "object"),
        )

    train_yolo(
        config=config,
        epochs_override=epochs,
        resume_path=str(resume) if resume else None,
        prepare_if_missing=prepare_dataset,
        copy_images=copy_images,
        eval_after_train=eval_after_train,
        eval_split=eval_split,
        eval_output_json=eval_output_json,
    )


if __name__ == "__main__":
    main()