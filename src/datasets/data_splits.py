from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from src.datasets.duplicate_groups import load_group_manifest
from src.datasets.ena24_dataset import ENA24DetectionDataset
from src.datasets.group_split import apply_split_to_samples, load_split_manifest
from src.datasets.split_validation import assert_full_coverage, assert_no_group_leakage

SplitStrategy = Literal["random", "manifest", "phash_groups"]

DEFAULT_METADATA_DIR = "metadata"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
GROUP_MANIFEST_FILENAME = "group_manifest.json"

PHASH_GROUPS_INSTRUCTION = (
    "Run the anti-leakage pipeline first:\n"
    "  python scripts/build_duplicate_groups.py --data_dir {data_dir}\n"
    "  python scripts/split_coco_by_groups.py --data_dir {data_dir} "
    "--group_manifest {data_dir}/metadata/group_manifest.json"
)


def split_kwargs_from_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "split_strategy": data_cfg.get("split_strategy", "random"),
        "split_manifest_path": data_cfg.get("split_manifest_path"),
        "group_manifest_path": data_cfg.get("group_manifest_path"),
    }


def _resolve_manifest_path(
    data_dir: Path,
    explicit_path: str | Path | None,
    default_filename: str,
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    return data_dir / DEFAULT_METADATA_DIR / default_filename


def _load_manifest_split(
    full_dataset: ENA24DetectionDataset,
    data_dir: Path,
    split_strategy: SplitStrategy,
    split_manifest_path: str | Path | None,
    group_manifest_path: str | Path | None,
) -> tuple[ENA24DetectionDataset, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if split_strategy == "manifest":
        if split_manifest_path is None:
            raise ValueError(
                "split_strategy='manifest' requires split_manifest_path in data config."
            )
        resolved_split_path = Path(split_manifest_path)
        resolved_group_path = (
            Path(group_manifest_path) if group_manifest_path is not None else None
        )
    elif split_strategy == "phash_groups":
        resolved_split_path = _resolve_manifest_path(
            data_dir=data_dir,
            explicit_path=split_manifest_path,
            default_filename=SPLIT_MANIFEST_FILENAME,
        )
        resolved_group_path = _resolve_manifest_path(
            data_dir=data_dir,
            explicit_path=group_manifest_path,
            default_filename=GROUP_MANIFEST_FILENAME,
        )
    else:
        raise ValueError(f"Unsupported manifest split strategy: {split_strategy}")

    if not resolved_split_path.exists():
        raise FileNotFoundError(
            f"split_manifest not found: {resolved_split_path}\n"
            + PHASH_GROUPS_INSTRUCTION.format(data_dir=data_dir)
        )

    split_manifest = load_split_manifest(resolved_split_path)
    group_manifest = None

    if resolved_group_path is not None:
        if not resolved_group_path.exists():
            raise FileNotFoundError(
                f"group_manifest not found: {resolved_group_path}\n"
                + PHASH_GROUPS_INSTRUCTION.format(data_dir=data_dir)
            )
        group_manifest = load_group_manifest(resolved_group_path)

    all_files = [sample["file_name"] for sample in full_dataset.samples]
    assert_full_coverage(all_files, split_manifest)

    if group_manifest is not None:
        assert_full_coverage(all_files, split_manifest, group_manifest)
        assert_no_group_leakage(group_manifest, split_manifest)

    train_samples, val_samples, test_samples = apply_split_to_samples(
        full_dataset.samples,
        split_manifest,
    )

    return full_dataset, train_samples, val_samples, test_samples


def prepare_data_splits(
    data_dir: str | Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    size: int | None = None,
    split_strategy: SplitStrategy | str = "random",
    split_manifest_path: str | Path | None = None,
    group_manifest_path: str | Path | None = None,
) -> tuple[ENA24DetectionDataset, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    strategy: SplitStrategy = split_strategy  # type: ignore[assignment]

    full_dataset = ENA24DetectionDataset(data_dir=data_dir)

    if strategy in ("manifest", "phash_groups"):
        if size is not None:
            print(
                f"Note: size is ignored when split_strategy='{strategy}' - "
                "using split_manifest instead."
            )

        return _load_manifest_split(
            full_dataset=full_dataset,
            data_dir=data_dir,
            split_strategy=strategy,
            split_manifest_path=split_manifest_path,
            group_manifest_path=group_manifest_path,
        )

    if strategy != "random":
        raise ValueError(
            f"Unknown split_strategy: {split_strategy!r}. "
            "Allowed: 'random', 'manifest', 'phash_groups'."
        )

    torch.manual_seed(seed)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()

    if size is not None:
        size = min(size, len(indices))
        indices = indices[:size]

    train_size = int(train_ratio * len(indices))
    val_size = int(val_ratio * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    test_samples = [full_dataset.samples[i] for i in test_indices]

    return full_dataset, train_samples, val_samples, test_samples


def prepare_data_splits_from_data_config(
    data_cfg: dict[str, Any],
    *,
    data_dir: str | Path,
    seed: int,
) -> tuple[ENA24DetectionDataset, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_kwargs = split_kwargs_from_data_config(data_cfg)

    return prepare_data_splits(
        data_dir=data_dir,
        train_ratio=data_cfg.get("train_ratio", 0.6),
        val_ratio=data_cfg.get("val_ratio", 0.2),
        seed=seed,
        size=data_cfg.get("size"),
        **split_kwargs,
    )
