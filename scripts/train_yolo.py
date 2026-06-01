#!/usr/bin/env python3
"""
Train Ultralytics YOLO on a prepared ENA24 YOLO dataset.

Default: src/config/yolo_config.json (sample, 20 images).
Full run:  --config_path src/config/yolo_config_full.json
           (prepare that dataset first; see README)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_yolo_dataset import prepare_yolo_dataset
from src.config.load_config import load_config
from src.config.paths import normalize_config_paths, resolve_data_yaml_path, resolve_project_path


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


def ensure_dataset_ready(config: dict, prepare_if_missing: bool, copy_images: bool) -> Path:
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

    print("Preparing YOLO dataset (data.yaml missing)...")
    prepare_yolo_dataset(
        coco_dir=Path(data_cfg["coco_dir"]),
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
):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required for YOLO training. Install with: pip install ultralytics"
        ) from error

    data_yaml_path = ensure_dataset_ready(
        config=config,
        prepare_if_missing=prepare_if_missing,
        copy_images=copy_images,
    )

    model_cfg = config["model"]
    training_cfg = config["training"]
    checkpoints_cfg = config["checkpoints"]

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

    results = model.train(**train_kwargs)

    run_save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(model.trainer.save_dir)
    weights_dir = run_save_dir / "weights"

    export_best_weights(
        run_weights_dir=weights_dir,
        destination_path=checkpoints_cfg["best_checkpoint_path"],
    )

    print("Training run directory:", run_save_dir)
    return results


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "yolo_config.json"),
    show_default=True,
    help="Sample: yolo_config.json | Full: yolo_config_full.json",
)
@click.option("--epochs", type=int, default=None, help="Override training.epochs from config.")
@click.option(
    "--resume",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Resume from a YOLO checkpoint (.pt), e.g. runs/detect/.../weights/last.pt",
)
@click.option(
    "--prepare",
    "prepare_dataset",
    is_flag=True,
    default=False,
    help="Run dataset preparation before training (or if data.yaml is missing).",
)
@click.option(
    "--copy_images",
    is_flag=True,
    default=False,
    help="Copy images during --prepare (for Windows without symlinks).",
)
def main(
    config_path: str,
    epochs: int | None,
    resume: str | None,
    prepare_dataset: bool,
    copy_images: bool,
):
    config = normalize_config_paths(load_config(config_path))

    if prepare_dataset:
        data_cfg = config["data"]
        detection_cfg = config.get("detection", {})
        print("Preparing YOLO dataset...")
        prepare_yolo_dataset(
            coco_dir=Path(data_cfg["coco_dir"]),
            output_dir=Path(data_cfg["output_dir"]),
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
    )


if __name__ == "__main__":
    main()
