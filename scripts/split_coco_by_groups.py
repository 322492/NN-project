#!/usr/bin/env python3
"""
Splits the ENA24 dataset into train/val/test at the near-duplicate group level.

Example:
  python scripts/split_coco_by_groups.py \\
    --data_dir data/ena24_full \\
    --group_manifest data/ena24_full/metadata/group_manifest.json \\
    --seed 42 --train_ratio 0.6 --val_ratio 0.2 \\
    --output_dir data/ena24_full/metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.coco_io import (
    list_available_images,
    load_coco_from_data_dir,
    load_json,
    save_json,
)
from src.datasets.duplicate_groups import load_group_manifest
from src.datasets.group_split import (
    export_coco_splits,
    save_split_manifest,
    split_groups,
)
from src.datasets.pipeline_metadata import (
    DEFAULT_SPLIT_SEED,
    METADATA_DIR_NAME,
    PIPELINE_CONFIG_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_STATISTICS_FILENAME,
    build_artifact_map,
    save_pipeline_config,
)
from src.datasets.split_validation import (
    compute_split_statistics,
    run_all_split_checks,
)

SPLITS_SUBDIR_NAME = "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates a group-aware split manifest for ENA24.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset directory (images/ + COCO annotations).",
    )
    parser.add_argument(
        "--group_manifest",
        type=Path,
        required=True,
        help="Path to group_manifest.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: <data_dir>/metadata).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Seed for shuffling groups (default: {DEFAULT_SPLIT_SEED}).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Fraction of groups assigned to train.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of groups assigned to val.",
    )
    parser.add_argument(
        "--export_coco_splits",
        action="store_true",
        help="Export train/val/test annotations.json to the splits/ subdirectory.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or (data_dir / METADATA_DIR_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco_from_data_dir(data_dir)
    available_images = list_available_images(coco, data_dir)
    all_files = sorted(Path(image["file_name"]).name for image in available_images)

    if not all_files:
        raise SystemExit(f"No available images in directory: {data_dir}")

    group_manifest = load_group_manifest(args.group_manifest)
    missing_groups = sorted(set(all_files) - set(group_manifest.keys()))
    if missing_groups:
        raise SystemExit(
            f"group_manifest does not cover {len(missing_groups)} images from the dataset. "
            f"Example: {missing_groups[0]}"
        )

    split_manifest = split_groups(
        group_manifest=group_manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    coco_splits = None
    if args.export_coco_splits:
        splits_dir = output_dir / SPLITS_SUBDIR_NAME
        exported_paths = export_coco_splits(coco, split_manifest, splits_dir)
        coco_splits = {
            split_name: load_json(path)
            for split_name, path in exported_paths.items()
        }
        log(f"Exported COCO annotations to: {splits_dir}")

    run_all_split_checks(
        all_files=all_files,
        group_manifest=group_manifest,
        split_manifest=split_manifest,
        coco_splits=coco_splits,
    )

    split_metadata = {
        "data_dir": str(data_dir),
        "group_manifest_path": str(args.group_manifest.resolve()),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }

    split_manifest_path = output_dir / SPLIT_MANIFEST_FILENAME
    save_split_manifest(split_manifest, split_manifest_path, metadata=split_metadata)

    statistics = compute_split_statistics(
        coco=coco,
        group_manifest=group_manifest,
        split_manifest=split_manifest,
    )
    statistics["split_metadata"] = split_metadata

    statistics_path = output_dir / SPLIT_STATISTICS_FILENAME
    save_json(statistics, statistics_path)

    save_pipeline_config(
        output_dir / PIPELINE_CONFIG_FILENAME,
        {
            "data_dir": str(data_dir),
            "split_seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "group_manifest_path": str(args.group_manifest.resolve()),
            "split_manifest_path": str(split_manifest_path),
            "split_statistics_path": str(statistics_path),
            "artifacts": build_artifact_map(output_dir),
        },
    )

    log("")
    log("Split created successfully:")
    log(f"  Images train/val/test: {statistics['images_per_split']}")
    log(f"  Groups train/val/test:  {statistics['groups_per_split']}")
    log(f"  Leakage check:         {statistics['leakage_check_passed']}")
    log(f"Saved: {split_manifest_path}")


if __name__ == "__main__":
    main()
