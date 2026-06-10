#!/usr/bin/env python3
"""
Builds group_manifest.json from pHash (imagehash) near-duplicate detection.

Example:
  python scripts/build_duplicate_groups.py \\
    --data_dir data/ena24_full \\
    --max_distance_threshold 10 \\
    --top_crop_frac 0.06 \\
    --bottom_crop_frac 0.10 \\
    --output_dir data/ena24_full/metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.coco_io import (
    list_available_images,
    load_coco_from_data_dir,
    resolve_image_path,
    save_json,
)
from src.datasets.duplicate_groups import (
    build_group_manifest,
    compute_phash_encodings,
    find_similar_pairs,
    load_encodings_cache,
    save_duplicate_pairs,
    save_encodings_cache,
    save_group_manifest,
    summarize_groups,
)
from src.datasets.hash_crop import hash_config_from_fractions, save_hash_config
from src.datasets.pipeline_metadata import (
    DEFAULT_MAX_GROUP_SIZE,
    DEFAULT_MAX_DISTANCE_THRESHOLD,
    DUPLICATE_PAIRS_FILENAME,
    GROUP_MANIFEST_FILENAME,
    GROUP_STATS_FILENAME,
    HASH_CONFIG_FILENAME,
    METADATA_DIR_NAME,
    PHASH_CACHE_FILENAME,
    PIPELINE_CONFIG_FILENAME,
    PIPELINE_VERSION,
    build_artifact_map,
    save_pipeline_config,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Groups ENA24 near-duplicate images using pHash.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset directory (images/ + COCO annotations).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=f"Output directory for manifests (default: <data_dir>/{METADATA_DIR_NAME}).",
    )
    parser.add_argument(
        "--max_distance_threshold",
        type=int,
        default=DEFAULT_MAX_DISTANCE_THRESHOLD,
        help=f"Hamming threshold for duplicate pairs (default: {DEFAULT_MAX_DISTANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--top_crop_frac",
        type=float,
        default=0.06,
        help="Fraction of height cropped from the top before hashing.",
    )
    parser.add_argument(
        "--bottom_crop_frac",
        type=float,
        default=0.10,
        help="Fraction of height cropped from the bottom before hashing.",
    )
    parser.add_argument(
        "--left_crop_frac",
        type=float,
        default=0.0,
        help="Fraction of width cropped from the left before hashing.",
    )
    parser.add_argument(
        "--right_crop_frac",
        type=float,
        default=0.0,
        help="Fraction of width cropped from the right before hashing.",
    )
    parser.add_argument(
        "--max_group_size",
        type=int,
        default=DEFAULT_MAX_GROUP_SIZE,
        help=(
            f"Max group size (Union-Find with cap; 0 = no limit). "
            f"Default: {DEFAULT_MAX_GROUP_SIZE} - tuned for typical ENA24 bursts (~50-60 frames)."
        ),
    )
    parser.add_argument(
        "--recompute_hashes",
        action="store_true",
        help="Force recomputation of pHash (ignore cache).",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def collect_image_paths(data_dir: Path, coco: dict[str, Any]) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}

    for image in list_available_images(coco, data_dir):
        file_name = Path(image["file_name"]).name
        image_path = resolve_image_path(data_dir, file_name)
        if image_path is not None:
            image_paths[file_name] = image_path

    return image_paths


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or (data_dir / METADATA_DIR_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_config = hash_config_from_fractions(
        top_frac=args.top_crop_frac,
        bottom_frac=args.bottom_crop_frac,
        left_frac=args.left_crop_frac,
        right_frac=args.right_crop_frac,
    )

    hash_config_path = output_dir / HASH_CONFIG_FILENAME
    save_hash_config(crop_config, hash_config_path)

    coco = load_coco_from_data_dir(data_dir)
    image_paths = collect_image_paths(data_dir, coco)

    if not image_paths:
        raise SystemExit(f"No available images in directory: {data_dir}")

    log(f"Found {len(image_paths)} images to hash.")

    cache_path = output_dir / PHASH_CACHE_FILENAME
    encodings = None if args.recompute_hashes else load_encodings_cache(cache_path)

    if encodings is None:
        log("Computing pHash (imagehash)...")
        encodings = compute_phash_encodings(
            image_paths=image_paths,
            crop_config=crop_config,
            show_progress=True,
        )
        encoding_metadata = {
            "data_dir": str(data_dir),
            "hash_backend": "imagehash",
            "hash_config": crop_config.to_dict(),
        }
        save_encodings_cache(encodings, cache_path, metadata=encoding_metadata)
        log(f"Saved hash cache: {cache_path}")
    else:
        missing_files = sorted(set(image_paths.keys()) - set(encodings.keys()))
        if missing_files:
            log(
                f"Incomplete cache ({len(missing_files)} missing files) - "
                "recomputing all hashes."
            )
            encodings = compute_phash_encodings(
                image_paths=image_paths,
                crop_config=crop_config,
                show_progress=True,
            )
            save_encodings_cache(
                encodings,
                cache_path,
                metadata={
                    "data_dir": str(data_dir),
                    "hash_backend": "imagehash",
                    "hash_config": crop_config.to_dict(),
                },
            )
        else:
            log(f"Loaded hash cache: {cache_path}")

    max_group_size = args.max_group_size if args.max_group_size > 0 else None
    cap_label = max_group_size if max_group_size is not None else "none"

    log(
        f"Finding duplicate pairs (threshold <= {args.max_distance_threshold}, "
        f"max_group_size={cap_label})..."
    )
    similar_pairs = find_similar_pairs(
        encodings=encodings,
        max_distance_threshold=args.max_distance_threshold,
    )
    pairs = [(left, right) for left, right, _distance in similar_pairs]

    pairs_path = output_dir / DUPLICATE_PAIRS_FILENAME
    save_duplicate_pairs(pairs, pairs_path)

    all_files = sorted(image_paths.keys())
    group_manifest = build_group_manifest(
        all_files=all_files,
        pairs=[],
        similar_pairs=similar_pairs,
        max_group_size=max_group_size,
    )
    group_stats = summarize_groups(group_manifest)

    metadata = {
        "data_dir": str(data_dir),
        "hash_backend": "imagehash",
        "max_distance_threshold": args.max_distance_threshold,
        "max_group_size": max_group_size,
        "hash_config_path": str(hash_config_path),
        "hash_config": crop_config.to_dict(),
        "num_duplicate_pairs": len(pairs),
        "group_statistics": group_stats,
    }

    group_manifest_path = output_dir / GROUP_MANIFEST_FILENAME
    save_group_manifest(group_manifest, group_manifest_path, metadata=metadata)

    stats_path = output_dir / GROUP_STATS_FILENAME
    save_json(group_stats, stats_path)

    save_pipeline_config(
        output_dir / PIPELINE_CONFIG_FILENAME,
        {
            "data_dir": str(data_dir),
            "hash_backend": "imagehash",
            "max_distance_threshold": args.max_distance_threshold,
            "hash_config": crop_config.to_dict(),
            "hash_config_path": str(hash_config_path),
            "group_build": {
                "num_duplicate_pairs": len(pairs),
                "group_statistics": group_stats,
                "max_group_size": max_group_size,
            },
            "artifacts": build_artifact_map(output_dir),
        },
    )

    log("")
    log("Group summary:")
    log(f"  Images:                  {group_stats['total_images']}")
    log(f"  Groups:                  {group_stats['num_groups']}")
    log(f"  Singleton groups:        {group_stats['num_singleton_groups']}")
    log(f"  Largest group:           {group_stats['largest_group_size']}")
    log(f"  % in groups >1 image:    {group_stats['pct_in_multi_image_groups']:.2f}")
    log(f"  Duplicate pairs:         {len(pairs)}")
    log("")
    log(f"Saved: {group_manifest_path}")
    log(f"Pipeline version: {PIPELINE_VERSION}")


if __name__ == "__main__":
    main()
