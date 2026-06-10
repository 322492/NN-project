"""Shared COCO annotation I/O helpers for ENA24."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datasets.ena24_dataset import xywh_to_xyxy

COCO_META_KEYS = ("info", "licenses", "categories")

ANNOTATIONS_CANDIDATES = (
    "annotations.json",
    "ena24_public.json",
    "ena24.json",
)


def load_coco_annotations(path: Path | str) -> dict[str, Any]:
    annotations_path = Path(path)
    with annotations_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_annotations_file(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)

    candidates = [
        data_dir / name for name in ANNOTATIONS_CANDIDATES
    ] + [
        data_dir / "metadata" / "ena24_public.json",
        data_dir / "metadata" / "ena24.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"COCO annotations file not found in directory: {data_dir}")


def load_coco_from_data_dir(data_dir: Path | str) -> dict[str, Any]:
    return load_coco_annotations(find_annotations_file(data_dir))


def normalize_image_id(image_id: Any) -> str:
    return str(image_id)


def build_image_index(coco: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        normalize_image_id(image["id"]): image
        for image in coco.get("images", [])
        if isinstance(image, dict) and "id" in image
    }


def build_annotations_by_image_id(
    coco: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    annotations_by_image_id: dict[str, list[dict[str, Any]]] = {}

    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict) or "image_id" not in annotation:
            continue

        image_id = normalize_image_id(annotation["image_id"])
        annotations_by_image_id.setdefault(image_id, []).append(annotation)

    return annotations_by_image_id


def filter_coco_by_image_ids(
    coco: dict[str, Any],
    image_ids: set[str] | list[str],
) -> dict[str, Any]:
    selected_ids = {normalize_image_id(image_id) for image_id in image_ids}

    subset: dict[str, Any] = {}
    for key in COCO_META_KEYS:
        if key in coco:
            subset[key] = coco[key]

    subset["images"] = [
        image
        for image in coco.get("images", [])
        if isinstance(image, dict)
        and normalize_image_id(image.get("id")) in selected_ids
    ]

    subset["annotations"] = [
        annotation
        for annotation in coco.get("annotations", [])
        if isinstance(annotation, dict)
        and normalize_image_id(annotation.get("image_id")) in selected_ids
    ]

    return subset


def resolve_image_path(data_dir: Path | str, file_name: str) -> Path | None:
    data_dir = Path(data_dir)
    normalized_name = Path(file_name).name

    candidates = [
        data_dir / "images" / normalized_name,
        data_dir / normalized_name,
        data_dir / "ENA24" / "images" / normalized_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def list_available_images(coco: dict[str, Any], data_dir: Path | str) -> list[dict[str, Any]]:
    available_images = []

    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue

        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue

        if resolve_image_path(data_dir, file_name) is None:
            continue

        available_images.append(image)

    return available_images


def samples_from_coco(
    data_dir: Path | str,
    coco: dict[str, Any] | None = None,
    only_existing: bool = True,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)

    if coco is None:
        coco = load_coco_from_data_dir(data_dir)

    annotations_by_image_id = build_annotations_by_image_id(coco)
    samples: list[dict[str, Any]] = []

    for image in coco.get("images", []):
        if not isinstance(image, dict):
            continue

        image_id = normalize_image_id(image["id"])
        file_name = Path(image["file_name"]).name
        image_path = resolve_image_path(data_dir, file_name)

        if only_existing and image_path is None:
            continue

        if image_path is None:
            image_path = data_dir / "images" / file_name

        bboxes = [
            xywh_to_xyxy(annotation["bbox"])
            for annotation in annotations_by_image_id.get(image_id, [])
            if "bbox" in annotation
        ]

        samples.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "width": image["width"],
                "height": image["height"],
                "bboxes": bboxes,
            }
        )

    return samples


def save_json(data: dict[str, Any] | list[Any], path: Path | str, indent: int = 2) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def load_json(path: Path | str) -> dict[str, Any] | list[Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
