#!/usr/bin/env python3
"""
Sweep production pHash parameters on the full (or large) dataset.

Uses the existing phash_encodings.json cache - only thresholds are recomputed quickly.

Example:
  python scripts/sweep_phash_production_params.py --data_dir data/ena24_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.coco_io import save_json
from src.datasets.duplicate_groups import (
    build_group_manifest,
    load_encodings_cache,
    find_similar_pairs,
)
from src.datasets.hash_crop import hash_config_from_fractions, load_hash_config
from src.datasets.phash_tuning import (
    PRODUCTION_THRESHOLD_SWEEP,
    export_cluster_previews,
    recommend_production_params,
    sweep_production_grid,
    sweep_thresholds,
)

DEFAULT_MAX_GROUP_SIZES = (40, 50, 60)
from src.datasets.pipeline_metadata import METADATA_DIR_NAME, PHASH_CACHE_FILENAME

DEFAULT_RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment: select Hamming threshold for production group-split.",
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--metadata_dir",
        type=Path,
        default=None,
        help="Default: <data_dir>/metadata",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Hamming thresholds (default 3-6 after sweeping 3-12).",
    )
    parser.add_argument(
        "--max_group_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_GROUP_SIZES),
        help="Union-Find group size limits (default 40 50 60 - ENA24 burst).",
    )
    parser.add_argument(
        "--no_max_group_size",
        action="store_true",
        help="Sweep threshold only (legacy behavior, no group size limit).",
    )
    parser.add_argument(
        "--export_previews",
        action="store_true",
        help="Export previews for the recommended threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "phash_production_sweep.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars (phase logs remain).",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def load_encodings(metadata_dir: Path) -> dict[str, str]:
    cache_path = metadata_dir / PHASH_CACHE_FILENAME
    encodings = load_encodings_cache(cache_path)
    if encodings is None:
        raise FileNotFoundError(
            f"Missing hash cache: {cache_path}\n"
            "Run first: python scripts/build_duplicate_groups.py --data_dir ..."
        )
    return encodings


def collect_image_paths_from_encodings(
    data_dir: Path,
    encodings: dict[str, str],
) -> dict[str, Path]:
    from src.datasets.coco_io import resolve_image_path

    image_paths: dict[str, Path] = {}
    for file_name in encodings:
        resolved = resolve_image_path(data_dir, file_name)
        if resolved is not None:
            image_paths[file_name] = resolved
    return image_paths


def print_results_table(ranked: list[dict[str, Any]]) -> None:
    log("")
    log("thr | cap | groups | multi% | largest | >20 | pairs | prod OK | score")
    log("-" * 80)
    for row in ranked:
        cap = row.get("max_group_size")
        cap_label = "none" if cap is None else str(cap)
        log(
            f"{row['max_distance_threshold']:>3} | "
            f"{cap_label:>4} | "
            f"{row['num_groups']:>6} | "
            f"{row['pct_in_multi_image_groups']:>6.1f} | "
            f"{row['largest_group_size']:>7} | "
            f"{row['groups_size_over_20']:>3} | "
            f"{row['num_duplicate_pairs']:>5} | "
            f"{'yes' if row['passes_production_criteria'] else 'no ':>7} | "
            f"{row['production_score']:>7.1f}"
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    metadata_dir = (args.metadata_dir or (data_dir / METADATA_DIR_NAME)).resolve()
    show_progress = not args.quiet

    log("=== Phase A: production threshold sweep (hash cache) ===")
    log(f"  data_dir     = {data_dir}")
    log(f"  metadata_dir = {metadata_dir}")
    log("")

    log("[1/4] Loading phash_encodings.json cache...")
    encodings = load_encodings(metadata_dir)
    log(f"      OK - {len(encodings)} hashes (no re-hashing).")

    hash_config_path = metadata_dir / "hash_config.json"
    if hash_config_path.exists():
        crop = load_hash_config(hash_config_path)
        log(f"      hash_config from cache: {crop.to_dict()}")
    else:
        crop = hash_config_from_fractions()
        log(f"      missing hash_config.json - default crop: {crop.to_dict()}")

    num_pairs_estimate = len(encodings) * (len(encodings) - 1) // 2
    log("")
    if args.no_max_group_size:
        log(
            f"[2/4] Sweeping {len(args.thresholds)} thresholds "
            f"({min(args.thresholds)}-{max(args.thresholds)}), no group size limit..."
        )
        sweep_results = sweep_thresholds(
            encodings,
            args.thresholds,
            show_progress=show_progress,
        )
    else:
        num_combos = len(args.thresholds) * len(args.max_group_sizes)
        log(
            f"[2/4] Sweeping {num_combos} combinations: "
            f"thresholds {args.thresholds} x max_group_size {args.max_group_sizes}..."
        )
        log(
            f"      ~{num_pairs_estimate:,} comparisons per threshold "
            f"(pair cache - no re-hashing)."
        )
        sweep_results = sweep_production_grid(
            encodings,
            args.thresholds,
            args.max_group_sizes,
            show_progress=show_progress,
        )

    if show_progress:
        log("      Progress: bar = thr x cap combinations.")
    log("")

    log("")
    log("[3/4] Ranking and production recommendation...")
    recommendation = recommend_production_params(sweep_results)

    print_results_table(recommendation["ranked"])

    best = recommendation["recommended"]
    log("")
    log("=== Production recommendation ===")
    log(f"  max_distance_threshold = {best['max_distance_threshold']}")
    log(f"  max_group_size         = {best.get('max_group_size', 'none')}")
    log(f"  largest_group_size     = {best['largest_group_size']}")
    log(f"  pct in groups >1       = {best['pct_in_multi_image_groups']:.2f}%")
    log(f"  num_groups             = {best['num_groups']}")
    log(f"  passes criteria        = {best['passes_production_criteria']}")
    if best.get("violations"):
        log(f"  violations             = {best['violations']}")

    passing_count = len(recommendation["all_passing"])
    log(f"  acceptable thresholds    = {passing_count} / {len(sweep_results)}")
    if passing_count == 0:
        log("")
        log(
            "  WARNING: no threshold meets prod OK=yes criteria. "
            "Look for rows with the smallest largest and pct_multi in the 10-50% range."
        )
    else:
        passing_thresholds = [
            item["max_distance_threshold"] for item in recommendation["all_passing"]
        ]
        log(f"  thresholds prod OK=yes   = {passing_thresholds}")

    payload = {
        "data_dir": str(data_dir),
        "metadata_dir": str(metadata_dir),
        "num_images": len(encodings),
        "hash_config": crop.to_dict(),
        "thresholds": args.thresholds,
        "max_group_sizes": None if args.no_max_group_size else args.max_group_sizes,
        "criteria": recommendation["criteria"],
        "sweep_results": recommendation["ranked"],
        "recommended": best,
        "all_passing_thresholds": [
            item["max_distance_threshold"] for item in recommendation["all_passing"]
        ],
    }

    log("")
    log("[4/4] Saving results...")
    if args.export_previews and passing_count > 0:
        threshold = best["max_distance_threshold"]
        cap = best.get("max_group_size")
        cap_suffix = f"_cap{cap}" if cap is not None else ""
        log(f"      Exporting previews for thr={threshold}{cap_suffix}...")
        image_paths = collect_image_paths_from_encodings(data_dir, encodings)
        similar_pairs = find_similar_pairs(
            encodings,
            threshold,
            show_progress=show_progress,
        )
        group_manifest = build_group_manifest(
            sorted(encodings.keys()),
            pairs=[],
            similar_pairs=similar_pairs,
            max_group_size=cap,
        )
        preview_dir = metadata_dir / "cluster_previews" / f"threshold_{threshold}{cap_suffix}"
        previews = export_cluster_previews(
            image_paths=image_paths,
            group_manifest=group_manifest,
            output_dir=preview_dir,
            num_clusters=20,
            show_progress=show_progress,
        )
        payload["preview_manifest"] = previews
        log(f"      Saved {len(previews)} previews: {preview_dir}")
    elif args.export_previews:
        log("      Skipping preview export - no threshold with prod OK=yes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output)
    log(f"      JSON report: {args.output.resolve()}")
    log("")
    log("Next step (after choosing a threshold):")
    cap_arg = best.get("max_group_size")
    cap_cmd = f" --max_group_size {cap_arg}" if cap_arg is not None else ""
    log(
        f"  python scripts/build_duplicate_groups.py "
        f"--data_dir {data_dir} "
        f"--max_distance_threshold {best['max_distance_threshold']}{cap_cmd}"
    )


if __name__ == "__main__":
    main()
