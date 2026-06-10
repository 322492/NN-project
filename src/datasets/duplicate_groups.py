"""Near-duplicate detection and stable group_id assignment (pHash via imagehash)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image
from tqdm import tqdm

from src.datasets.coco_io import load_json, save_json
from src.datasets.hash_crop import HashCropConfig, crop_for_hashing_with_config
from src.datasets.pipeline_metadata import PIPELINE_VERSION, unwrap_versioned_payload


class UnionFind:
    def __init__(self, items: list[str], max_group_size: int | None = None):
        self.parent = {item: item for item in items}
        self.component_size = {item: 1 for item in items}
        self.max_group_size = max_group_size

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def _size(self, root: str) -> int:
        return self.component_size[root]

    def union(self, left: str, right: str) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return False

        size_left = self._size(root_left)
        size_right = self._size(root_right)

        if self.max_group_size is not None and size_left + size_right > self.max_group_size:
            return False

        if root_left < root_right:
            self.parent[root_right] = root_left
            self.component_size[root_left] = size_left + size_right
        else:
            self.parent[root_left] = root_right
            self.component_size[root_right] = size_left + size_right

        return True


def _normalize_file_name(file_name: str) -> str:
    return Path(file_name).name


def _encode_phash(image: Image.Image) -> str:
    return str(imagehash.phash(image))


def encode_image_for_hash(
    image_path: Path | str,
    crop_config: HashCropConfig | dict[str, Any] | None = None,
) -> str:
    image = Image.open(image_path).convert("RGB")
    cropped = crop_for_hashing_with_config(image, crop_config)
    return _encode_phash(cropped)


def compute_phash_encodings(
    image_paths: dict[str, Path | str],
    crop_config: HashCropConfig | dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, str]:
    """Compute pHash encodings. image_paths maps file_name -> file path."""
    encodings: dict[str, str] = {}
    items = sorted(image_paths.items())
    iterator = tqdm(items, desc="Computing pHash") if show_progress else items

    for file_name, image_path in iterator:
        encodings[file_name] = encode_image_for_hash(
            image_path=image_path,
            crop_config=crop_config,
        )

    return encodings


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def find_similar_pairs(
    encodings: dict[str, str],
    max_distance_threshold: int,
    *,
    show_progress: bool = False,
) -> list[tuple[str, str, int]]:
    """Similar image pairs with Hamming distance, sorted ascending."""
    file_names = sorted(encodings.keys())
    pairs: list[tuple[str, str, int]] = []

    outer_range = range(len(file_names))
    if show_progress:
        outer_range = tqdm(
            outer_range,
            desc=f"Pair comparisons (thr={max_distance_threshold})",
            leave=False,
            unit="img",
        )

    for index_left in outer_range:
        for index_right in range(index_left + 1, len(file_names)):
            left_name = file_names[index_left]
            right_name = file_names[index_right]
            distance = hamming_distance(encodings[left_name], encodings[right_name])

            if distance <= max_distance_threshold:
                pairs.append((left_name, right_name, distance))

    pairs.sort(key=lambda item: item[2])
    return pairs


def find_duplicate_pairs(
    encodings: dict[str, str],
    max_distance_threshold: int,
    *,
    show_progress: bool = False,
) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left, right, _distance in find_similar_pairs(
            encodings,
            max_distance_threshold,
            show_progress=show_progress,
        )
    ]


def build_groups_from_pairs(
    all_files: list[str],
    pairs: list[tuple[str, str]],
    *,
    pair_distances: list[int] | None = None,
    max_group_size: int | None = None,
) -> dict[str, str]:
    normalized_files = sorted({_normalize_file_name(file_name) for file_name in all_files})
    union_find = UnionFind(normalized_files, max_group_size=max_group_size)

    ordered_pairs = list(pairs)
    if pair_distances is not None:
        if len(pair_distances) != len(pairs):
            raise ValueError("pair_distances must have the same length as pairs.")
        ordered_pairs = [
            pair
            for _, pair in sorted(
                zip(pair_distances, pairs, strict=True),
                key=lambda item: item[0],
            )
        ]

    for left_name, right_name in ordered_pairs:
        union_find.union(_normalize_file_name(left_name), _normalize_file_name(right_name))

    components: dict[str, list[str]] = defaultdict(list)
    for file_name in normalized_files:
        root = union_find.find(file_name)
        components[root].append(file_name)

    file_to_group: dict[str, str] = {}
    for group_index, root in enumerate(sorted(components.keys()), start=1):
        group_id = f"group_{group_index:06d}"
        for file_name in sorted(components[root]):
            file_to_group[file_name] = group_id

    return file_to_group


def assign_singleton_groups(
    all_files: list[str],
    file_to_group: dict[str, str] | None = None,
) -> dict[str, str]:
    result = dict(file_to_group or {})
    next_index = 1

    if result:
        existing_numbers = [
            int(group_id.split("_")[1])
            for group_id in set(result.values())
            if group_id.startswith("group_")
        ]
        if existing_numbers:
            next_index = max(existing_numbers) + 1

    for file_name in sorted({_normalize_file_name(name) for name in all_files}):
        if file_name not in result:
            result[file_name] = f"group_{next_index:06d}"
            next_index += 1

    return result


def build_group_manifest(
    all_files: list[str],
    pairs: list[tuple[str, str]],
    *,
    similar_pairs: list[tuple[str, str, int]] | None = None,
    max_group_size: int | None = None,
) -> dict[str, str]:
    if similar_pairs is not None:
        pair_list = [(left, right) for left, right, _distance in similar_pairs]
        distances = [distance for _left, _right, distance in similar_pairs]
        grouped = build_groups_from_pairs(
            all_files,
            pair_list,
            pair_distances=distances,
            max_group_size=max_group_size,
        )
    else:
        grouped = build_groups_from_pairs(
            all_files,
            pairs,
            max_group_size=max_group_size,
        )

    return assign_singleton_groups(all_files, grouped)


def summarize_groups(group_manifest: dict[str, str]) -> dict[str, Any]:
    group_sizes = Counter(group_manifest.values())
    sizes = list(group_sizes.values())

    return {
        "total_images": len(group_manifest),
        "num_groups": len(group_sizes),
        "num_singleton_groups": sum(1 for size in sizes if size == 1),
        "largest_group_size": max(sizes) if sizes else 0,
        "pct_in_multi_image_groups": (
            100.0 * sum(size for size in sizes if size > 1) / len(group_manifest)
            if group_manifest
            else 0.0
        ),
    }


def load_encodings_cache(path: Path | str) -> dict[str, str] | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None

    payload = load_json(cache_path)
    if isinstance(payload, dict) and "encodings" in payload:
        encodings = payload["encodings"]
    else:
        encodings = unwrap_versioned_payload(payload)

    if not isinstance(encodings, dict):
        raise ValueError(f"Invalid phash_encodings format: {cache_path}")

    return encodings


def save_encodings_cache(
    encodings: dict[str, str],
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    document = {
        "pipeline_version": PIPELINE_VERSION,
        "encodings": encodings,
    }
    if metadata:
        document["metadata"] = metadata

    save_json(document, path)


def save_duplicate_pairs(pairs: list[tuple[str, str]], path: Path | str) -> None:
    serializable = [{"file_a": left, "file_b": right} for left, right in pairs]
    save_json(serializable, path)


def load_duplicate_pairs(path: Path | str) -> list[tuple[str, str]]:
    raw_pairs = load_json(path)
    return [(item["file_a"], item["file_b"]) for item in raw_pairs]


def save_group_manifest(
    group_manifest: dict[str, str],
    path: Path | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "groups": group_manifest,
    }
    if metadata:
        payload["metadata"] = metadata

    save_json(payload, path)


def load_group_manifest(path: Path | str) -> dict[str, str]:
    payload = load_json(path)

    if isinstance(payload, dict) and "groups" in payload:
        return payload["groups"]

    if isinstance(payload, dict):
        return payload

    raise ValueError(f"Invalid group_manifest format: {path}")
