"""Reproducibility metadata for the anti-leakage pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.datasets.coco_io import load_json, save_json

PIPELINE_VERSION = "1.0.0"
DEFAULT_SPLIT_SEED = 42
DEFAULT_MAX_DISTANCE_THRESHOLD = 10
DEFAULT_MAX_GROUP_SIZE = 50  # typical ENA24 burst: ~50-60 frames

METADATA_DIR_NAME = "metadata"
HASH_CONFIG_FILENAME = "hash_config.json"
PHASH_CACHE_FILENAME = "phash_encodings.json"
DUPLICATE_PAIRS_FILENAME = "duplicate_pairs.json"
GROUP_MANIFEST_FILENAME = "group_manifest.json"
GROUP_STATS_FILENAME = "group_statistics.json"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
SPLIT_STATISTICS_FILENAME = "split_statistics.json"
PIPELINE_CONFIG_FILENAME = "pipeline_config.json"
THRESHOLD_SWEEP_FILENAME = "threshold_sweep.json"
CROP_COMPARISON_FILENAME = "crop_comparison.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def metadata_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / METADATA_DIR_NAME


def artifact_path(data_dir: Path | str, filename: str) -> Path:
    return metadata_dir(data_dir) / filename


def load_pipeline_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    payload = load_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid pipeline_config format: {config_path}")

    return payload


def save_pipeline_config(
    path: Path | str,
    updates: dict[str, Any],
    *,
    merge: bool = True,
) -> dict[str, Any]:
    config_path = Path(path)
    existing = load_pipeline_config(config_path) if merge and config_path.exists() else {}

    payload = {
        **existing,
        **updates,
        "pipeline_version": PIPELINE_VERSION,
        "updated_at": utc_now_iso(),
    }

    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]

    save_json(payload, config_path)
    return payload


def build_artifact_map(output_dir: Path | str) -> dict[str, str]:
    output_dir = Path(output_dir)
    artifacts = {
        "hash_config": str(output_dir / HASH_CONFIG_FILENAME),
        "phash_encodings": str(output_dir / PHASH_CACHE_FILENAME),
        "duplicate_pairs": str(output_dir / DUPLICATE_PAIRS_FILENAME),
        "group_manifest": str(output_dir / GROUP_MANIFEST_FILENAME),
        "group_statistics": str(output_dir / GROUP_STATS_FILENAME),
        "split_manifest": str(output_dir / SPLIT_MANIFEST_FILENAME),
        "split_statistics": str(output_dir / SPLIT_STATISTICS_FILENAME),
        "pipeline_config": str(output_dir / PIPELINE_CONFIG_FILENAME),
        "threshold_sweep": str(output_dir / THRESHOLD_SWEEP_FILENAME),
        "crop_comparison": str(output_dir / CROP_COMPARISON_FILENAME),
    }
    return {name: path for name, path in artifacts.items() if Path(path).exists()}


def unwrap_versioned_payload(payload: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(payload, dict) and "data" in payload and "pipeline_version" in payload:
        return payload["data"]
    return payload
