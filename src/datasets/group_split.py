"""Train/val/test split at near-duplicate group level."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from src.datasets.coco_io import (
    filter_coco_by_image_ids,
    load_json,
    normalize_image_id,
    save_json,
)
from src.datasets.pipeline_metadata import PIPELINE_VERSION

SplitName = Literal["train", "val", "test"]
VALID_SPLITS = ("train", "val", "test")


def _group_by_id(group_manifest: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)

    for file_name, group_id in group_manifest.items():
        groups[group_id].append(file_name)

    for group_id in groups:
        groups[group_id].sort()

    return dict(groups)


def split_groups(
    group_manifest: dict[str, str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, SplitName]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Required: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1.0")

    groups = _group_by_id(group_manifest)
    group_ids = sorted(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_groups = len(group_ids)
    train_count = max(1, int(train_ratio * total_groups)) if total_groups > 0 else 0
    val_count = int(val_ratio * total_groups)

    if train_count + val_count >= total_groups:
        val_count = max(0, total_groups - train_count - 1)

    train_group_ids = set(group_ids[:train_count])
    val_group_ids = set(group_ids[train_count : train_count + val_count])
    test_group_ids = set(group_ids[train_count + val_count :])

    split_manifest: dict[str, SplitName] = {}

    for group_id, file_names in groups.items():
        if group_id in train_group_ids:
            split_name: SplitName = "train"
        elif group_id in val_group_ids:
            split_name = "val"
        else:
            split_name = "test"

        for file_name in file_names:
            split_manifest[file_name] = split_name

    return split_manifest


def apply_split_to_samples(
    samples: list[dict[str, Any]],
    split_manifest: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []

    for sample in samples:
        file_name = Path(sample["file_name"]).name
        split_name = split_manifest.get(file_name)

        if split_name == "train":
            train_samples.append(sample)
        elif split_name == "val":
            val_samples.append(sample)
        elif split_name == "test":
            test_samples.append(sample)
        else:
            raise KeyError(f"No split assignment for image: {file_name}")

    return train_samples, val_samples, test_samples


def image_ids_for_split(
    coco: dict[str, Any],
    split_manifest: dict[str, str],
    split_name: SplitName,
) -> set[str]:
    file_names = {
        file_name
        for file_name, assigned_split in split_manifest.items()
        if assigned_split == split_name
    }

    image_ids: set[str] = set()
    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue

        file_name = Path(image.get("file_name", "")).name
        if file_name in file_names:
            image_ids.add(normalize_image_id(image["id"]))

    return image_ids


def export_coco_splits(
    coco: dict[str, Any],
    split_manifest: dict[str, str],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}

    for split_name in VALID_SPLITS:
        image_ids = image_ids_for_split(coco, split_manifest, split_name)
        split_coco = filter_coco_by_image_ids(coco, image_ids)

        output_path = output_dir / f"{split_name}_annotations.json"
        save_json(split_coco, output_path)
        exported_paths[split_name] = output_path

    return exported_paths


def downsample_train_by_group(
    split_manifest: dict[str, str],
    group_manifest: dict[str, str],
    seed: int = 42,
    max_images_per_group: int = 1,
) -> dict[str, SplitName]:
    if max_images_per_group < 1:
        raise ValueError("max_images_per_group must be >= 1")

    groups = _group_by_id(group_manifest)
    rng = random.Random(seed)

    updated_manifest = dict(split_manifest)
    train_files = [
        file_name
        for file_name, split_name in split_manifest.items()
        if split_name == "train"
    ]

    files_by_group: dict[str, list[str]] = defaultdict(list)
    for file_name in train_files:
        files_by_group[group_manifest[file_name]].append(file_name)

    for group_id, file_names in files_by_group.items():
        if len(file_names) <= max_images_per_group:
            continue

        selected = sorted(file_names)
        rng.shuffle(selected)
        keep = set(selected[:max_images_per_group])

        for file_name in file_names:
            if file_name not in keep:
                del updated_manifest[file_name]

    return updated_manifest


def save_split_manifest(
    split_manifest: dict[str, str],
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "splits": split_manifest,
    }
    if metadata:
        payload["metadata"] = metadata

    save_json(payload, path)


def load_split_manifest(path: Path | str) -> dict[str, str]:
    payload = load_json(path)

    if isinstance(payload, dict) and "splits" in payload:
        return payload["splits"]

    if isinstance(payload, dict):
        return payload

    raise ValueError(f"Invalid split_manifest format: {path}")
