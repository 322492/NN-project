"""Sanity checks and statistics for group-aware splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.coco_io import build_annotations_by_image_id, build_image_index, normalize_image_id


class SplitValidationError(ValueError):
    pass


def assert_disjoint_splits(split_manifest: dict[str, str]) -> None:
    if len(split_manifest) != len(set(split_manifest.keys())):
        raise SplitValidationError("split_manifest contains duplicate file_name entries.")


def assert_no_group_leakage(
    group_manifest: dict[str, str],
    split_manifest: dict[str, str],
) -> None:
    group_to_splits: dict[str, set[str]] = defaultdict(set)

    for file_name, split_name in split_manifest.items():
        if file_name not in group_manifest:
            raise SplitValidationError(f"Missing group_id for file: {file_name}")

        group_id = group_manifest[file_name]
        group_to_splits[group_id].add(split_name)

    leaking_groups = {
        group_id: sorted(split_names)
        for group_id, split_names in group_to_splits.items()
        if len(split_names) > 1
    }

    if leaking_groups:
        example = next(iter(leaking_groups.items()))
        raise SplitValidationError(
            "Group leakage detected across splits. "
            f"Example: {example[0]} -> {example[1]}"
        )


def assert_full_coverage(
    all_files: list[str],
    split_manifest: dict[str, str],
    group_manifest: dict[str, str] | None = None,
) -> None:
    normalized_files = {Path(file_name).name for file_name in all_files}

    missing_splits = normalized_files - set(split_manifest.keys())
    if missing_splits:
        raise SplitValidationError(
            f"Missing split assignment for {len(missing_splits)} images."
        )

    if group_manifest is None:
        return

    missing_groups = normalized_files - set(group_manifest.keys())
    if missing_groups:
        raise SplitValidationError(
            f"Missing group_id for {len(missing_groups)} images."
        )


def assert_coco_consistency(coco_split: dict[str, Any]) -> None:
    image_index = build_image_index(coco_split)
    image_ids = set(image_index.keys())

    for annotation in coco_split.get("annotations", []):
        if not isinstance(annotation, dict):
            continue

        image_id = normalize_image_id(annotation.get("image_id"))
        if image_id not in image_ids:
            raise SplitValidationError(
                f"Annotation references missing image_id: {image_id}"
            )


def run_all_split_checks(
    all_files: list[str],
    group_manifest: dict[str, str],
    split_manifest: dict[str, str],
    coco_splits: dict[str, dict[str, Any]] | None = None,
) -> None:
    assert_disjoint_splits(split_manifest)
    assert_full_coverage(all_files, split_manifest, group_manifest)
    assert_no_group_leakage(group_manifest, split_manifest)

    if coco_splits:
        for split_name, coco_split in coco_splits.items():
            try:
                assert_coco_consistency(coco_split)
            except SplitValidationError as exc:
                raise SplitValidationError(f"[{split_name}] {exc}") from exc


def _class_distribution_for_split(
    coco: dict[str, Any],
    split_manifest: dict[str, str],
    split_name: str,
) -> dict[str, int]:
    file_names = {
        file_name
        for file_name, assigned_split in split_manifest.items()
        if assigned_split == split_name
    }

    image_index = build_image_index(coco)
    file_name_to_image_id = {
        Path(image["file_name"]).name: image_id
        for image_id, image in image_index.items()
    }

    selected_image_ids = {
        file_name_to_image_id[file_name]
        for file_name in file_names
        if file_name in file_name_to_image_id
    }

    annotations_by_image_id = build_annotations_by_image_id(coco)
    class_counter: Counter[str] = Counter()

    for image_id in selected_image_ids:
        for annotation in annotations_by_image_id.get(image_id, []):
            category_id = annotation.get("category_id", "unknown")
            class_counter[str(category_id)] += 1

    return dict(sorted(class_counter.items(), key=lambda item: item[0]))


def compute_split_statistics(
    coco: dict[str, Any],
    group_manifest: dict[str, str],
    split_manifest: dict[str, str],
) -> dict[str, Any]:
    group_sizes = Counter(group_manifest.values())
    sizes = list(group_sizes.values())

    images_per_split = Counter(split_manifest.values())
    groups_per_split: dict[str, set[str]] = defaultdict(set)

    for file_name, split_name in split_manifest.items():
        groups_per_split[split_name].add(group_manifest[file_name])

    annotations_per_split = {
        split_name: sum(_class_distribution_for_split(coco, split_manifest, split_name).values())
        for split_name in ("train", "val", "test")
    }

    class_distribution_per_split = {
        split_name: _class_distribution_for_split(coco, split_manifest, split_name)
        for split_name in ("train", "val", "test")
    }

    leakage_check_passed = True
    try:
        assert_no_group_leakage(group_manifest, split_manifest)
    except SplitValidationError:
        leakage_check_passed = False

    return {
        "total_images": len(split_manifest),
        "num_groups": len(group_sizes),
        "num_singleton_groups": sum(1 for size in sizes if size == 1),
        "largest_group_size": max(sizes) if sizes else 0,
        "pct_in_multi_image_groups": (
            100.0 * sum(size for size in sizes if size > 1) / len(group_manifest)
            if group_manifest
            else 0.0
        ),
        "images_per_split": {
            split_name: images_per_split.get(split_name, 0)
            for split_name in ("train", "val", "test")
        },
        "annotations_per_split": annotations_per_split,
        "groups_per_split": {
            split_name: len(groups_per_split.get(split_name, set()))
            for split_name in ("train", "val", "test")
        },
        "class_distribution_per_split": class_distribution_per_split,
        "leakage_check_passed": leakage_check_passed,
    }


def count_random_split_leakage(
    group_manifest: dict[str, str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_trials: int = 1,
) -> dict[str, Any]:
    leakages = []

    for trial_offset in range(num_trials):
        trial_result = simulate_naive_image_split_leakage(
            group_manifest=group_manifest,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + trial_offset,
        )
        leakages.append(trial_result["leaking_groups_count"])

    return {
        "num_trials": num_trials,
        "leaking_groups_per_trial": leakages,
        "mean_leaking_groups": sum(leakages) / len(leakages) if leakages else 0.0,
    }


def simulate_naive_image_split_leakage(
    group_manifest: dict[str, str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    file_names = sorted(group_manifest.keys())
    rng = random.Random(seed)
    rng.shuffle(file_names)

    train_size = int(train_ratio * len(file_names))
    val_size = int(val_ratio * len(file_names))

    naive_split = {}
    for index, file_name in enumerate(file_names):
        if index < train_size:
            naive_split[file_name] = "train"
        elif index < train_size + val_size:
            naive_split[file_name] = "val"
        else:
            naive_split[file_name] = "test"

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    for file_name, split_name in naive_split.items():
        group_to_splits[group_manifest[file_name]].add(split_name)

    leaking_groups = {
        group_id: sorted(split_names)
        for group_id, split_names in group_to_splits.items()
        if len(split_names) > 1
    }

    return {
        "leaking_groups_count": len(leaking_groups),
        "leaking_groups_pct": (
            100.0 * len(leaking_groups) / len(set(group_manifest.values()))
            if group_manifest
            else 0.0
        ),
        "example_leaking_groups": dict(list(leaking_groups.items())[:5]),
    }
