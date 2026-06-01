#!/usr/bin/env python3
"""
Convert ENA24 COCO annotations to Ultralytics YOLO layout.

Single-class mode (default): every box is class 0 — localization only, aligned with
the baseline binary window classifier. Species labels from COCO are ignored.

Output:
  <output_dir>/images/{train,val,test}/
  <output_dir>/labels/{train,val,test}/
  <output_dir>/data.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.config.paths import normalize_config_paths, resolve_project_path
from src.datasets.data_splits import prepare_data_splits

SINGLE_CLASS_ID = 0


def load_coco_annotations(coco_dir: Path) -> dict:
    annotations_path = coco_dir / "annotations.json"
    with annotations_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_labels_by_image_id(
    annotations: list[dict],
    single_class: bool,
    coco_id_to_yolo: dict[int, int] | None = None,
) -> dict[str, list[tuple[int, list[float]]]]:
    labels_by_image_id: dict[str, list[tuple[int, list[float]]]] = {}

    for annotation in annotations:
        image_id = str(annotation["image_id"])
        bbox_xywh = annotation["bbox"]

        if single_class:
            yolo_class_id = SINGLE_CLASS_ID
        else:
            category_id = annotation["category_id"]
            if coco_id_to_yolo is None or category_id not in coco_id_to_yolo:
                raise ValueError(f"Unknown category_id {category_id} for image_id {image_id}")
            yolo_class_id = coco_id_to_yolo[category_id]

        if image_id not in labels_by_image_id:
            labels_by_image_id[image_id] = []

        labels_by_image_id[image_id].append((yolo_class_id, bbox_xywh))

    return labels_by_image_id


def build_multi_class_maps(categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    sorted_categories = sorted(categories, key=lambda category: category["id"])
    coco_id_to_yolo = {
        category["id"]: index for index, category in enumerate(sorted_categories)
    }
    class_names = [category["name"] for category in sorted_categories]
    return coco_id_to_yolo, class_names


def coco_bbox_to_yolo_line(
    bbox_xywh: list[float],
    image_width: int,
    image_height: int,
    class_id: int,
) -> str:
    x, y, box_width, box_height = bbox_xywh

    x_center = (x + box_width / 2) / image_width
    y_center = (y + box_height / 2) / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    norm_width = min(max(norm_width, 0.0), 1.0)
    norm_height = min(max(norm_height, 0.0), 1.0)

    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
    )


def link_or_copy_image(source_path: Path, destination_path: Path, copy_images: bool) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        return

    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    if copy_images:
        shutil.copy2(source_path, destination_path)
        return

    try:
        destination_path.symlink_to(source_path.resolve())
    except OSError:
        shutil.copy2(source_path, destination_path)


def write_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    labels: list[tuple[int, list[float]]],
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        coco_bbox_to_yolo_line(bbox_xywh, image_width, image_height, class_id)
        for class_id, bbox_xywh in labels
    ]

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_split(
    split_name: str,
    samples: list[dict],
    labels_by_image_id: dict[str, list[tuple[int, list[float]]]],
    images_by_id: dict[str, dict],
    output_dir: Path,
    copy_images: bool,
) -> int:
    exported = 0

    for sample in samples:
        image_id = str(sample["image_id"])
        image_info = images_by_id.get(image_id)

        if image_info is None:
            continue

        file_name = Path(image_info["file_name"]).name
        image_width = int(image_info["width"])
        image_height = int(image_info["height"])

        source_image_path = Path(sample["image_path"])
        destination_image_path = output_dir / "images" / split_name / file_name
        label_path = output_dir / "labels" / split_name / f"{Path(file_name).stem}.txt"

        link_or_copy_image(source_image_path, destination_image_path, copy_images=copy_images)

        image_labels = labels_by_image_id.get(image_id, [])
        write_label_file(label_path, image_width, image_height, image_labels)
        exported += 1

    return exported


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    data_yaml_path = output_dir / "data.yaml"
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False, default_flow_style=False)

    return data_yaml_path


def prepare_yolo_dataset(
    coco_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    size: int | None = None,
    copy_images: bool = False,
    single_class: bool = True,
    class_name: str = "object",
) -> dict:
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        labels_split_dir = output_dir / "labels" / split_name
        if labels_split_dir.exists():
            shutil.rmtree(labels_split_dir)

    coco_data = load_coco_annotations(coco_dir)

    if single_class:
        class_names = [class_name]
        coco_id_to_yolo = None
    else:
        coco_id_to_yolo, class_names = build_multi_class_maps(coco_data["categories"])

    labels_by_image_id = build_labels_by_image_id(
        coco_data["annotations"],
        single_class=single_class,
        coco_id_to_yolo=coco_id_to_yolo,
    )

    images_by_id = {str(image["id"]): image for image in coco_data["images"]}

    _, train_samples, val_samples, test_samples = prepare_data_splits(
        data_dir=str(coco_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        size=size,
    )

    train_count = export_split(
        "train", train_samples, labels_by_image_id, images_by_id, output_dir, copy_images
    )
    val_count = export_split(
        "val", val_samples, labels_by_image_id, images_by_id, output_dir, copy_images
    )
    test_count = export_split(
        "test", test_samples, labels_by_image_id, images_by_id, output_dir, copy_images
    )

    data_yaml_path = write_data_yaml(output_dir, class_names)

    summary = {
        "coco_dir": str(coco_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_yaml": str(data_yaml_path.resolve()),
        "seed": seed,
        "size": size,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "single_class": single_class,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_images": train_count,
        "val_images": val_count,
        "test_images": test_count,
        "copy_images": copy_images,
    }

    manifest_path = output_dir / "split_summary.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "yolo_config.json"),
    show_default=True,
)
@click.option("--coco_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--output_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--size", type=int, default=None, help="Limit images (same as baseline data.size).")
@click.option(
    "--copy_images",
    is_flag=True,
    default=False,
    help="Copy images instead of symlinks (use if symlinks fail on Windows).",
)
@click.option(
    "--multi_class",
    is_flag=True,
    default=False,
    help="Use all COCO species as separate classes (off by default).",
)
def main(
    config_path: str,
    coco_dir: Path | None,
    output_dir: Path | None,
    seed: int | None,
    size: int | None,
    copy_images: bool,
    multi_class: bool,
):
    config = normalize_config_paths(load_config(config_path))

    data_cfg = config["data"]
    detection_cfg = config.get("detection", {})

    single_class = not multi_class
    if multi_class:
        single_class = False
    elif "single_class" in detection_cfg:
        single_class = bool(detection_cfg["single_class"])

    class_name = detection_cfg.get("class_name", "object")

    resolved_coco_dir = resolve_project_path(coco_dir or data_cfg["coco_dir"])
    resolved_output_dir = resolve_project_path(output_dir or data_cfg["output_dir"])
    resolved_seed = seed if seed is not None else config["seed"]
    resolved_size = size if size is not None else data_cfg.get("size")

    print("Config:", config_path)
    print("COCO dir:", resolved_coco_dir)
    print("Output dir:", resolved_output_dir)
    print("Seed:", resolved_seed)
    print("Size limit:", resolved_size)
    print("Single class (boxes only):", single_class)
    if single_class:
        print("Class name:", class_name)
    resolved_copy_images = copy_images or bool(data_cfg.get("copy_images", False))
    print("Copy images:", resolved_copy_images)

    summary = prepare_yolo_dataset(
        coco_dir=resolved_coco_dir,
        output_dir=resolved_output_dir,
        seed=resolved_seed,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        size=resolved_size,
        copy_images=resolved_copy_images,
        single_class=single_class,
        class_name=class_name,
    )

    print()
    print("YOLO dataset ready.")
    print("data.yaml:", summary["data_yaml"])
    print("train:", summary["train_images"], "| val:", summary["val_images"], "| test:", summary["test_images"])
    print("classes:", summary["num_classes"], summary["class_names"])
    print("summary:", resolved_output_dir / "split_summary.json")


if __name__ == "__main__":
    main()
