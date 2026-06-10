"""Hamming threshold and crop variant tuning for pHash grouping."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.datasets.duplicate_groups import (
    build_group_manifest,
    compute_phash_encodings,
    find_similar_pairs,
    summarize_groups,
)
from src.datasets.hash_crop import HashCropConfig, hash_config_from_fractions

DEFAULT_THRESHOLD_SWEEP = (4, 6, 8, 10, 12, 14)
PRODUCTION_THRESHOLD_SWEEP = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_SAMPLE_SIZE = 800
DEFAULT_PREVIEW_CLUSTERS = 20
MAX_RECOMMENDED_CLUSTER_SIZE = 50
DEFAULT_RECOMMENDED_THRESHOLD = 10

PRODUCTION_MAX_LARGEST_GROUP = 50
PRODUCTION_MIN_PCT_MULTI = 10.0
PRODUCTION_MAX_PCT_MULTI = 50.0


def sample_image_paths(
    image_paths: dict[str, Path | str],
    sample_size: int,
    seed: int,
) -> dict[str, Path]:
    normalized_paths = {name: Path(path) for name, path in image_paths.items()}
    file_names = sorted(normalized_paths.keys())

    if len(file_names) <= sample_size:
        return normalized_paths

    rng = random.Random(seed)
    selected = sorted(rng.sample(file_names, sample_size))
    return {name: normalized_paths[name] for name in selected}


def _group_size_distribution(group_manifest: dict[str, str]) -> dict[str, int]:
    sizes = list(Counter(group_manifest.values()).values())
    return {
        "groups_size_2_5": sum(1 for size in sizes if 2 <= size <= 5),
        "groups_size_6_10": sum(1 for size in sizes if 6 <= size <= 10),
        "groups_size_11_20": sum(1 for size in sizes if 11 <= size <= 20),
        "groups_size_over_20": sum(1 for size in sizes if size > 20),
        "groups_size_over_50": sum(1 for size in sizes if size > 50),
    }


def evaluate_from_similar_pairs(
    all_files: list[str],
    similar_pairs: list[tuple[str, str, int]],
    max_distance_threshold: int,
    *,
    max_group_size: int | None = None,
) -> dict[str, Any]:
    filtered_pairs = [
        (left, right, distance)
        for left, right, distance in similar_pairs
        if distance <= max_distance_threshold
    ]
    group_manifest = build_group_manifest(
        all_files=all_files,
        pairs=[],
        similar_pairs=filtered_pairs,
        max_group_size=max_group_size,
    )
    stats = summarize_groups(group_manifest)
    num_multi_image_groups = stats["num_groups"] - stats["num_singleton_groups"]

    return {
        "max_distance_threshold": max_distance_threshold,
        "max_group_size": max_group_size,
        "num_duplicate_pairs": len(filtered_pairs),
        "num_multi_image_groups": num_multi_image_groups,
        **_group_size_distribution(group_manifest),
        **stats,
        "passes_cluster_size_check": stats["largest_group_size"] <= MAX_RECOMMENDED_CLUSTER_SIZE,
    }


def evaluate_threshold(
    encodings: dict[str, str],
    max_distance_threshold: int,
    *,
    max_group_size: int | None = None,
    similar_pairs: list[tuple[str, str, int]] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    all_files = sorted(encodings.keys())

    if similar_pairs is None:
        similar_pairs = find_similar_pairs(
            encodings=encodings,
            max_distance_threshold=max_distance_threshold,
            show_progress=show_progress,
        )

    return evaluate_from_similar_pairs(
        all_files,
        similar_pairs,
        max_distance_threshold,
        max_group_size=max_group_size,
    )


def evaluate_production_candidate(result: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []

    if result["largest_group_size"] > PRODUCTION_MAX_LARGEST_GROUP:
        violations.append(
            f"largest_group_size={result['largest_group_size']} > {PRODUCTION_MAX_LARGEST_GROUP}"
        )
    if result["pct_in_multi_image_groups"] < PRODUCTION_MIN_PCT_MULTI:
        violations.append(
            f"pct_multi={result['pct_in_multi_image_groups']:.1f} < {PRODUCTION_MIN_PCT_MULTI}"
        )
    if result["pct_in_multi_image_groups"] > PRODUCTION_MAX_PCT_MULTI:
        violations.append(
            f"pct_multi={result['pct_in_multi_image_groups']:.1f} > {PRODUCTION_MAX_PCT_MULTI}"
        )
    if result["num_multi_image_groups"] < 1:
        violations.append("no multi-image groups")

    passes = len(violations) == 0
    distance_from_sweet_spot = abs(result["pct_in_multi_image_groups"] - 30.0)

    score = (
        (0 if passes else -1000)
        - result["largest_group_size"] * 10
        - result["groups_size_over_20"] * 50
        - result["groups_size_over_50"] * 500
        - distance_from_sweet_spot
        + result["num_multi_image_groups"] * 0.1
    )

    return {
        "passes_production_criteria": passes,
        "violations": violations,
        "production_score": score,
    }


def recommend_production_params(
    sweep_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not sweep_results:
        raise ValueError("No sweep results to rank.")

    ranked: list[dict[str, Any]] = []
    for result in sweep_results:
        evaluation = evaluate_production_candidate(result)
        ranked.append({**result, **evaluation})

    ranked.sort(key=lambda item: item["production_score"], reverse=True)
    passing = [item for item in ranked if item["passes_production_criteria"]]

    return {
        "criteria": {
            "max_largest_group": PRODUCTION_MAX_LARGEST_GROUP,
            "pct_multi_range": [PRODUCTION_MIN_PCT_MULTI, PRODUCTION_MAX_PCT_MULTI],
            "notes": (
                "Production: no clusters >50, moderate burst coverage (10-50%), "
                "prefer smallest largest_group_size with acceptable pct_multi."
            ),
        },
        "recommended": passing[0] if passing else ranked[0],
        "all_passing": passing,
        "ranked": ranked,
    }


def sweep_thresholds(
    encodings: dict[str, str],
    thresholds: list[int] | tuple[int, ...],
    *,
    max_group_size: int | None = None,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    threshold_iter = thresholds
    if show_progress:
        threshold_iter = tqdm(
            thresholds,
            desc="Hamming threshold sweep",
            unit="thr",
        )

    for threshold in threshold_iter:
        result = evaluate_threshold(
            encodings,
            threshold,
            max_group_size=max_group_size,
            show_progress=show_progress,
        )
        results.append(result)
        if show_progress and hasattr(threshold_iter, "set_postfix"):
            threshold_iter.set_postfix(
                pairs=result["num_duplicate_pairs"],
                largest=result["largest_group_size"],
                refresh=False,
            )

    return results


def sweep_production_grid(
    encodings: dict[str, str],
    thresholds: list[int] | tuple[int, ...],
    max_group_sizes: list[int] | tuple[int, ...],
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    max_threshold = max(thresholds)
    all_files = sorted(encodings.keys())

    similar_pairs = find_similar_pairs(
        encodings=encodings,
        max_distance_threshold=max_threshold,
        show_progress=show_progress,
    )

    combos = [(threshold, max_group_size) for threshold in thresholds for max_group_size in max_group_sizes]
    results: list[dict[str, Any]] = []
    combo_iter = combos

    if show_progress:
        combo_iter = tqdm(combos, desc="Sweep thr x max_group_size", unit="combo")

    for threshold, max_group_size in combo_iter:
        result = evaluate_from_similar_pairs(
            all_files,
            similar_pairs,
            threshold,
            max_group_size=max_group_size,
        )
        results.append(result)
        if show_progress and hasattr(combo_iter, "set_postfix"):
            combo_iter.set_postfix(
                thr=threshold,
                cap=max_group_size,
                largest=result["largest_group_size"],
                refresh=False,
            )

    return results


def recommend_threshold(
    sweep_results: list[dict[str, Any]],
    preferred_threshold: int = DEFAULT_RECOMMENDED_THRESHOLD,
) -> dict[str, Any]:
    if not sweep_results:
        raise ValueError("No sweep results for threshold recommendation.")

    passing = [result for result in sweep_results if result["passes_cluster_size_check"]]
    candidates = passing or sweep_results

    def sort_key(result: dict[str, Any]) -> tuple[int, float, int]:
        distance_from_preferred = abs(result["max_distance_threshold"] - preferred_threshold)
        return (
            distance_from_preferred,
            -result["pct_in_multi_image_groups"],
            result["largest_group_size"],
        )

    recommended = min(candidates, key=sort_key)

    return {
        "recommended_threshold": recommended["max_distance_threshold"],
        "preferred_threshold": preferred_threshold,
        "used_cluster_size_fallback": not bool(passing),
        "criteria": {
            "max_recommended_cluster_size": MAX_RECOMMENDED_CLUSTER_SIZE,
            "notes": (
                "Pick threshold with largest_group_size <= 50; "
                "on tie prefer max_distance_threshold=10."
            ),
        },
        "selected_result": recommended,
    }


def compare_crop_variants(
    image_paths: dict[str, Path | str],
    thresholds: list[int] | tuple[int, ...],
    cropped_config: HashCropConfig,
    no_crop_config: HashCropConfig | None = None,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    if no_crop_config is None:
        no_crop_config = hash_config_from_fractions(0.0, 0.0, 0.0, 0.0)

    variants = {
        "with_crop": cropped_config,
        "no_crop": no_crop_config,
    }

    comparison: dict[str, Any] = {}

    for variant_name, crop_config in variants.items():
        encodings = compute_phash_encodings(
            image_paths=image_paths,
            crop_config=crop_config,
            show_progress=show_progress,
        )
        sweep_results = sweep_thresholds(encodings, thresholds)
        recommendation = recommend_threshold(sweep_results)

        comparison[variant_name] = {
            "hash_config": crop_config.to_dict(),
            "threshold_sweep": sweep_results,
            "recommendation": recommendation,
        }

    recommended_variant = _recommend_crop_variant(comparison)
    comparison["recommended_variant"] = recommended_variant

    return comparison


def _recommend_crop_variant(comparison: dict[str, Any]) -> dict[str, Any]:
    with_crop = comparison["with_crop"]["recommendation"]["selected_result"]
    no_crop = comparison["no_crop"]["recommendation"]["selected_result"]

    with_crop_ok = with_crop["passes_cluster_size_check"]
    no_crop_ok = no_crop["passes_cluster_size_check"]

    if with_crop_ok and not no_crop_ok:
        choice = "with_crop"
    elif no_crop_ok and not with_crop_ok:
        choice = "no_crop"
    elif with_crop["largest_group_size"] < no_crop["largest_group_size"]:
        choice = "with_crop"
    elif no_crop["largest_group_size"] < with_crop["largest_group_size"]:
        choice = "no_crop"
    else:
        choice = (
            "with_crop"
            if with_crop["pct_in_multi_image_groups"] >= no_crop["pct_in_multi_image_groups"]
            else "no_crop"
        )

    return {
        "variant": choice,
        "hash_config": comparison[choice]["hash_config"],
        "recommended_threshold": comparison[choice]["recommendation"]["recommended_threshold"],
        "selected_result": comparison[choice]["recommendation"]["selected_result"],
    }


def export_cluster_previews(
    image_paths: dict[str, Path | str],
    group_manifest: dict[str, str],
    output_dir: Path | str,
    *,
    num_clusters: int = DEFAULT_PREVIEW_CLUSTERS,
    max_images_per_cluster: int = 6,
    thumbnail_size: int = 200,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[str]] = defaultdict(list)
    for file_name, group_id in group_manifest.items():
        groups[group_id].append(file_name)

    multi_image_groups = [
        (group_id, sorted(file_names))
        for group_id, file_names in groups.items()
        if len(file_names) > 1
    ]
    multi_image_groups.sort(key=lambda item: len(item[1]), reverse=True)

    previews: list[dict[str, Any]] = []

    cluster_iter = multi_image_groups[:num_clusters]
    if show_progress:
        cluster_iter = tqdm(
            cluster_iter,
            desc="Exporting cluster previews",
            unit="cluster",
        )

    for group_id, file_names in cluster_iter:
        selected_files = file_names[:max_images_per_cluster]
        thumbnails: list[Image.Image] = []

        for file_name in selected_files:
            image_path = image_paths.get(file_name)
            if image_path is None:
                continue

            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumbnail_size, thumbnail_size))
            thumbnails.append(image)

        if not thumbnails:
            continue

        canvas_width = sum(image.width for image in thumbnails)
        canvas_height = max(image.height for image in thumbnails)
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

        offset_x = 0
        for image in thumbnails:
            canvas.paste(image, (offset_x, 0))
            offset_x += image.width

        preview_path = output_dir / f"{group_id}.jpg"
        canvas.save(preview_path)

        previews.append(
            {
                "group_id": group_id,
                "group_size": len(file_names),
                "files": selected_files,
                "preview_path": str(preview_path.resolve()),
            }
        )

    return previews
