#!/usr/bin/env python3
"""
Reports group-aware split statistics (console + JSON).

Example:
  python scripts/report_split_statistics.py \\
    --data_dir data/ena24_full \\
    --group_manifest data/ena24_full/metadata/group_manifest.json \\
    --split_manifest data/ena24_full/metadata/split_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.coco_io import load_coco_from_data_dir, save_json
from src.datasets.duplicate_groups import load_group_manifest
from src.datasets.group_split import load_split_manifest
from src.datasets.split_validation import (
    compute_split_statistics,
    count_random_split_leakage,
)

DEFAULT_OUTPUT_FILENAME = "split_statistics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generates an ENA24 split statistics report.",
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
        "--split_manifest",
        type=Path,
        required=True,
        help="Path to split_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the JSON report.",
    )
    parser.add_argument(
        "--compare_random",
        action="store_true",
        help="Include a naive per-image split simulation.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="train_ratio used in the --compare_random simulation.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="val_ratio used in the --compare_random simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used in the --compare_random simulation.",
    )
    parser.add_argument(
        "--random_trials",
        type=int,
        default=1,
        help="Number of naive split simulation trials.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def print_statistics(statistics: dict) -> None:
    log("=== Split report ===")
    log(f"Total images:            {statistics['total_images']}")
    log(f"Groups:                  {statistics['num_groups']}")
    log(f"Singleton groups:        {statistics['num_singleton_groups']}")
    log(f"Largest group:           {statistics['largest_group_size']}")
    log(f"% in groups >1 image:    {statistics['pct_in_multi_image_groups']:.2f}")
    log("")
    log(f"Images per split:        {statistics['images_per_split']}")
    log(f"Annotations per split:   {statistics['annotations_per_split']}")
    log(f"Groups per split:        {statistics['groups_per_split']}")
    log(f"Leakage check passed:    {statistics['leakage_check_passed']}")

    if "random_split_comparison" in statistics:
        comparison = statistics["random_split_comparison"]
        log("")
        log("=== Comparison with naive per-image split ===")
        log(f"Trials:                  {comparison['num_trials']}")
        log(f"Mean leaking groups:     {comparison['mean_leaking_groups']:.2f}")
        log(f"Leaks per trial:         {comparison['leaking_groups_per_trial']}")


def main() -> None:
    args = parse_args()

    coco = load_coco_from_data_dir(args.data_dir)
    group_manifest = load_group_manifest(args.group_manifest)
    split_manifest = load_split_manifest(args.split_manifest)

    statistics = compute_split_statistics(
        coco=coco,
        group_manifest=group_manifest,
        split_manifest=split_manifest,
    )

    if args.compare_random:
        statistics["random_split_comparison"] = count_random_split_leakage(
            group_manifest=group_manifest,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_trials=args.random_trials,
        )

    print_statistics(statistics)

    output_path = args.output
    if output_path is not None:
        save_json(statistics, output_path)
        log(f"\nSaved report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
