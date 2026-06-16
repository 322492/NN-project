#!/usr/bin/env python3
"""
Exports sample images from train/val/test for visual inspection.

Example:
  python scripts/preview_split_samples.py --data_dir data/ena24_full
  python scripts/preview_split_samples.py --data_dir data/ena24_full --per_group --num_groups 10
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.datasets.coco_io import resolve_image_path, save_json
from src.datasets.duplicate_groups import load_group_manifest
from src.datasets.group_split import load_split_manifest
from src.datasets.phash_tuning import export_cluster_previews

DEFAULT_SAMPLES_PER_SPLIT = 30
DEFAULT_OUTPUT_DIR_NAME = "split_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export train/val/test image samples to preview folders.",
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="Default: <data_dir>/metadata/split_manifest.json",
    )
    parser.add_argument(
        "--group_manifest",
        type=Path,
        default=None,
        help="Optional, for previewing multi-image groups.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=f"Default: <data_dir>/metadata/{DEFAULT_OUTPUT_DIR_NAME}",
    )
    parser.add_argument(
        "--samples_per_split",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPLIT,
        help="Random number of images per split (train/val/test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks (recommended on Windows).",
    )
    parser.add_argument(
        "--per_group",
        action="store_true",
        help="Additionally: collages of the largest multi-image groups per split.",
    )
    parser.add_argument(
        "--num_groups",
        type=int,
        default=10,
        help="Number of largest groups to include in collages (with --per_group).",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def files_by_split(split_manifest: dict[str, str]) -> dict[str, list[str]]:
    by_split: dict[str, list[str]] = defaultdict(list)
    for file_name, split_name in split_manifest.items():
        by_split[split_name].append(file_name)
    return {name: sorted(files) for name, files in by_split.items()}


def export_random_samples(
    data_dir: Path,
    split_files: dict[str, list[str]],
    output_dir: Path,
    samples_per_split: int,
    seed: int,
    *,
    copy_files: bool,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    exported: dict[str, list[str]] = {}

    for split_name in ("train", "val", "test"):
        files = split_files.get(split_name, [])
        if not files:
            continue

        sample_count = min(samples_per_split, len(files))
        selected = sorted(rng.sample(files, sample_count))
        split_dir = output_dir / "samples" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for file_name in selected:
            source = resolve_image_path(data_dir, file_name)
            if source is None:
                continue

            destination = split_dir / file_name
            if copy_files:
                shutil.copy2(source, destination)
            else:
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                destination.symlink_to(source.resolve())

        exported[split_name] = selected
        log(f"  {split_name}: {len(selected)} images -> {split_dir}")

    return exported


def export_group_collages_per_split(
    data_dir: Path,
    split_manifest: dict[str, str],
    group_manifest: dict[str, str],
    output_dir: Path,
    num_groups: int,
) -> dict[str, list[dict]]:
    image_paths: dict[str, Path] = {}
    for file_name in split_manifest:
        resolved = resolve_image_path(data_dir, file_name)
        if resolved is not None:
            image_paths[file_name] = resolved

    groups_in_split: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for file_name, split_name in split_manifest.items():
        if file_name in group_manifest:
            groups_in_split[split_name][group_manifest[file_name]].append(file_name)

    result: dict[str, list[dict]] = {}

    for split_name in ("train", "val", "test"):
        multi_groups = [
            (group_id, sorted(file_names))
            for group_id, file_names in groups_in_split.get(split_name, {}).items()
            if len(file_names) > 1
        ]
        multi_groups.sort(key=lambda item: len(item[1]), reverse=True)

        if not multi_groups:
            continue

        subset_manifest = {
            file_name: group_id
            for group_id, file_names in multi_groups[:num_groups]
            for file_name in file_names
        }
        collage_dir = output_dir / "groups" / split_name
        previews = export_cluster_previews(
            image_paths=image_paths,
            group_manifest=subset_manifest,
            output_dir=collage_dir,
            num_clusters=num_groups,
        )
        result[split_name] = previews
        log(f"  {split_name}: {len(previews)} group collages -> {collage_dir}")

    return result


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    metadata_dir = data_dir / "metadata"
    split_manifest_path = args.split_manifest or (metadata_dir / "split_manifest.json")
    output_dir = args.output_dir or (metadata_dir / DEFAULT_OUTPUT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = load_split_manifest(split_manifest_path)
    split_files = files_by_split(split_manifest)

    log("Exporting random samples per split:")
    samples = export_random_samples(
        data_dir=data_dir,
        split_files=split_files,
        output_dir=output_dir,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
        copy_files=args.copy,
    )

    payload: dict = {
        "data_dir": str(data_dir),
        "split_manifest": str(split_manifest_path),
        "samples_per_split": args.samples_per_split,
        "seed": args.seed,
        "samples": samples,
        "counts": {name: len(files) for name, files in split_files.items()},
    }

    if args.per_group:
        group_manifest_path = args.group_manifest or (metadata_dir / "group_manifest.json")
        group_manifest = load_group_manifest(group_manifest_path)
        log("Exporting collages of the largest groups per split:")
        payload["group_previews"] = export_group_collages_per_split(
            data_dir=data_dir,
            split_manifest=split_manifest,
            group_manifest=group_manifest,
            output_dir=output_dir,
            num_groups=args.num_groups,
        )

    manifest_path = output_dir / "preview_manifest.json"
    save_json(payload, manifest_path)

    log("")
    log(f"Preview saved to: {output_dir}")
    log(f"  samples/train|val|test/  - random single images")
    if args.per_group:
        log(f"  groups/train|val|test/   - collages of the largest groups")
    log(f"  preview_manifest.json")
    log("")
    log("Open folders in Windows Explorer or run:")
    log(f"  explorer {output_dir / 'samples'}")


if __name__ == "__main__":
    main()
