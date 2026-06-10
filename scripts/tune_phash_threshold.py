#!/usr/bin/env python3
"""
Hamming threshold sweep and crop vs no-crop comparison on an image subset.

Example:
  python scripts/tune_phash_threshold.py \\
    --data_dir data/ena24_full \\
    --sample_size 800 \\
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
    find_duplicate_pairs,
)
from src.datasets.hash_crop import hash_config_from_fractions, save_hash_config
from src.datasets.phash_tuning import (
    DEFAULT_PREVIEW_CLUSTERS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_THRESHOLD_SWEEP,
    compare_crop_variants,
    export_cluster_previews,
    recommend_threshold,
    sample_image_paths,
    sweep_thresholds,
)
from src.datasets.pipeline_metadata import (
    CROP_COMPARISON_FILENAME,
    DEFAULT_MAX_DISTANCE_THRESHOLD,
    HASH_CONFIG_FILENAME,
    PIPELINE_CONFIG_FILENAME,
    THRESHOLD_SWEEP_FILENAME,
    build_artifact_map,
    save_pipeline_config,
)

DEFAULT_OUTPUT_DIR_NAME = "metadata"
PREVIEW_DIR_NAME = "cluster_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select max_distance_threshold and crop variant for pHash.",
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <data_dir>/metadata",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of images in the subset (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for subset sampling and preview export.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_SWEEP),
        help="List of Hamming thresholds to sweep.",
    )
    parser.add_argument(
        "--top_crop_frac",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--bottom_crop_frac",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--left_crop_frac",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--right_crop_frac",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--skip_crop_comparison",
        action="store_true",
        help="Skip crop vs no-crop comparison (faster sweep).",
    )
    parser.add_argument(
        "--export_previews",
        action="store_true",
        help="Save previews of ~20 largest clusters.",
    )
    parser.add_argument(
        "--num_preview_clusters",
        type=int,
        default=DEFAULT_PREVIEW_CLUSTERS,
    )
    parser.add_argument(
        "--write_hash_config",
        action="store_true",
        help="Write recommended hash_config.json to output_dir.",
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


def print_sweep_table(sweep_results: list[dict[str, Any]]) -> None:
    log("")
    log("threshold | groups | largest | % multi | pairs | ok")
    log("-" * 58)
    for result in sweep_results:
        log(
            f"{result['max_distance_threshold']:>9} | "
            f"{result['num_groups']:>6} | "
            f"{result['largest_group_size']:>7} | "
            f"{result['pct_in_multi_image_groups']:>7.2f} | "
            f"{result['num_duplicate_pairs']:>5} | "
            f"{'yes' if result['passes_cluster_size_check'] else 'no'}"
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or (data_dir / DEFAULT_OUTPUT_DIR_NAME)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_config = hash_config_from_fractions(
        top_frac=args.top_crop_frac,
        bottom_frac=args.bottom_crop_frac,
        left_frac=args.left_crop_frac,
        right_frac=args.right_crop_frac,
    )

    coco = load_coco_from_data_dir(data_dir)
    all_image_paths = collect_image_paths(data_dir, coco)
    if not all_image_paths:
        raise SystemExit(f"No available images in directory: {data_dir}")

    sampled_paths = sample_image_paths(
        image_paths=all_image_paths,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    log(f"Subset: {len(sampled_paths)} / {len(all_image_paths)} images.")

    if args.skip_crop_comparison:
        encodings = compute_phash_encodings(
            image_paths=sampled_paths,
            crop_config=cropped_config,
            show_progress=True,
        )
        sweep_results = sweep_thresholds(encodings, args.thresholds)
        recommendation = recommend_threshold(
            sweep_results,
            preferred_threshold=DEFAULT_MAX_DISTANCE_THRESHOLD,
        )
        crop_comparison = None
    else:
        log("Comparing crop vs no-crop...")
        crop_comparison = compare_crop_variants(
            image_paths=sampled_paths,
            thresholds=args.thresholds,
            cropped_config=cropped_config,
            show_progress=True,
        )
        recommended_variant = crop_comparison["recommended_variant"]
        sweep_results = crop_comparison[recommended_variant["variant"]]["threshold_sweep"]
        recommendation = crop_comparison[recommended_variant["variant"]]["recommendation"]
        recommendation["recommended_crop_variant"] = recommended_variant["variant"]
        recommendation["recommended_hash_config"] = recommended_variant["hash_config"]

    print_sweep_table(sweep_results)

    sweep_path = output_dir / THRESHOLD_SWEEP_FILENAME
    sweep_payload = {
        "data_dir": str(data_dir),
        "sample_size": len(sampled_paths),
        "seed": args.seed,
        "thresholds": args.thresholds,
        "hash_config": cropped_config.to_dict(),
        "results": sweep_results,
        "recommendation": recommendation,
    }
    save_json(sweep_payload, sweep_path)

    if crop_comparison is not None:
        crop_comparison_path = output_dir / CROP_COMPARISON_FILENAME
        save_json(crop_comparison, crop_comparison_path)
        log(f"Saved crop comparison: {crop_comparison_path}")

    recommended_threshold = recommendation["recommended_threshold"]
    recommended_hash_config = recommendation.get("recommended_hash_config", cropped_config.to_dict())

    if args.export_previews:
        encodings = compute_phash_encodings(
            image_paths=sampled_paths,
            crop_config=hash_config_from_fractions(
                top_frac=float(recommended_hash_config.get("top_frac", 0.06)),
                bottom_frac=float(recommended_hash_config.get("bottom_frac", 0.10)),
                left_frac=float(recommended_hash_config.get("left_frac", 0.0)),
                right_frac=float(recommended_hash_config.get("right_frac", 0.0)),
            ),
            show_progress=False,
        )
        pairs = find_duplicate_pairs(encodings, recommended_threshold)
        group_manifest = build_group_manifest(sorted(sampled_paths.keys()), pairs)
        preview_dir = output_dir / PREVIEW_DIR_NAME
        previews = export_cluster_previews(
            image_paths=sampled_paths,
            group_manifest=group_manifest,
            output_dir=preview_dir,
            num_clusters=args.num_preview_clusters,
        )
        save_json(previews, preview_dir / "preview_manifest.json")
        log(f"Saved {len(previews)} cluster previews: {preview_dir}")

    if args.write_hash_config:
        hash_config_path = output_dir / HASH_CONFIG_FILENAME
        save_hash_config(
            hash_config_from_fractions(**recommended_hash_config),
            hash_config_path,
        )
        log(f"Saved recommended hash_config: {hash_config_path}")

    save_pipeline_config(
        output_dir / PIPELINE_CONFIG_FILENAME,
        {
            "data_dir": str(data_dir),
            "tuning": {
                "sample_size": len(sampled_paths),
                "seed": args.seed,
                "thresholds": args.thresholds,
                "recommended_threshold": recommended_threshold,
                "recommended_hash_config": recommended_hash_config,
            },
            "artifacts": build_artifact_map(output_dir),
        },
    )

    log("")
    log("Recommendation:")
    log(f"  max_distance_threshold = {recommended_threshold}")
    if crop_comparison is not None:
        log(f"  crop_variant           = {recommendation.get('recommended_crop_variant')}")
    log(f"  largest_group_size     = {recommendation['selected_result']['largest_group_size']}")
    log(f"  % in groups >1         = {recommendation['selected_result']['pct_in_multi_image_groups']:.2f}")
    log(f"Saved sweep: {sweep_path}")
    log("")
    log(
        "Next step:\n"
        f"  python scripts/build_duplicate_groups.py "
        f"--data_dir {data_dir} --max_distance_threshold {recommended_threshold} "
        f"--output_dir {output_dir}"
    )


if __name__ == "__main__":
    main()
