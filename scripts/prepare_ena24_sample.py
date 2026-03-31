#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_SAMPLE_SIZE = 30
DEFAULT_SEED = 42

OUTPUT_DIR = Path("data/ena24_sample")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_ANNOTATIONS_PATH = OUTPUT_DIR / "annotations.json"

ENA24_INFO_URL = "https://lila.science/datasets/ena24detection/"
ENA24_METADATA_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24_public.json"
ENA24_IMAGE_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small ENA24 sample dataset for development."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Optional local ENA24 dataset directory. If omitted, files are fetched from the public ENA24 source.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of images to include in the sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for deterministic sampling (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message)


def prepare_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for path in OUTPUT_IMAGE_DIR.iterdir():
        if path.is_file():
            path.unlink()

    if OUTPUT_ANNOTATIONS_PATH.exists():
        OUTPUT_ANNOTATIONS_PATH.unlink()


def load_json_from_url(url: str) -> dict[str, Any]:
    with urlopen(url) as response:
        return json.load(response)


def load_json_from_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_local_annotations_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "ena24_public.json",
        data_dir / "ena24.json",
        data_dir / "annotations.json",
        data_dir / "metadata" / "ena24_public.json",
        data_dir / "metadata" / "ena24.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find an ENA24 annotation file in the provided directory.\n"
        "TODO: adjust candidate annotation paths in scripts/prepare_ena24_sample.py if your local layout differs."
    )


def find_local_image_path(data_dir: Path, file_name: str) -> Path | None:
    candidates = [
        data_dir / "images" / file_name,
        data_dir / file_name,
        data_dir / "ENA24" / "images" / file_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_annotations(data_dir: Path | None) -> tuple[dict[str, Any], str]:
    log("Loading annotations...")

    if data_dir is None:
        annotations = load_json_from_url(ENA24_METADATA_URL)
        return annotations, "public"

    annotations_path = find_local_annotations_file(data_dir)
    annotations = load_json_from_file(annotations_path)
    return annotations, "local"


def filter_usable_images(
    annotations: dict[str, Any],
    data_dir: Path | None,
    source_mode: str,
) -> list[dict[str, Any]]:
    image_records = annotations.get("images", [])
    usable_images = []

    for image_record in image_records:
        if not isinstance(image_record, dict):
            continue

        file_name = image_record.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            continue

        if source_mode == "local":
            assert data_dir is not None
            if find_local_image_path(data_dir, file_name) is None:
                continue

        usable_images.append(image_record)

    return usable_images


def sample_images(
    annotations: dict[str, Any],
    data_dir: Path | None,
    source_mode: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    log("Sampling images...")
    usable_images = filter_usable_images(annotations, data_dir, source_mode)

    if not usable_images:
        raise RuntimeError(
            "No usable images were found in the ENA24 annotations.\n"
            "TODO: adjust image path resolution if the dataset layout differs."
        )

    sample_count = min(sample_size, len(usable_images))
    rng = random.Random(seed)
    return rng.sample(usable_images, sample_count)


def build_annotation_subset(
    annotations: dict[str, Any],
    sampled_images: list[dict[str, Any]],
) -> dict[str, Any]:
    sampled_ids = {image["id"] for image in sampled_images if "id" in image}
    sampled_annotations = [
        annotation
        for annotation in annotations.get("annotations", [])
        if isinstance(annotation, dict) and annotation.get("image_id") in sampled_ids
    ]

    subset = {}
    for key in ("info", "licenses", "categories"):
        if key in annotations:
            subset[key] = annotations[key]

    subset["images"] = sampled_images
    subset["annotations"] = sampled_annotations
    return subset


def copy_local_images(data_dir: Path, sampled_images: list[dict[str, Any]]) -> int:
    saved = 0

    for image_record in sampled_images:
        file_name = image_record["file_name"]
        source_path = find_local_image_path(data_dir, file_name)
        if source_path is None:
            log(f"[warning] Could not find local image: {file_name}")
            continue

        destination = OUTPUT_IMAGE_DIR / Path(file_name).name
        shutil.copy2(source_path, destination)
        saved += 1

    return saved


def download_public_images(sampled_images: list[dict[str, Any]]) -> int:
    saved = 0

    for image_record in sampled_images:
        file_name = image_record["file_name"]
        image_url = f"{ENA24_IMAGE_BASE_URL}/{file_name}"
        destination = OUTPUT_IMAGE_DIR / Path(file_name).name

        try:
            with urlopen(image_url) as response, destination.open("wb") as target:
                shutil.copyfileobj(response, target)
            saved += 1
        except (HTTPError, URLError) as exc:
            log(f"[warning] Could not download image: {file_name} ({exc})")

    return saved


def save_sample(
    annotations: dict[str, Any],
    sampled_images: list[dict[str, Any]],
    data_dir: Path | None,
    source_mode: str,
) -> None:
    log("Saving sample...")
    prepare_output_directory()

    subset = build_annotation_subset(annotations, sampled_images)
    with OUTPUT_ANNOTATIONS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2)

    if source_mode == "local":
        assert data_dir is not None
        saved_images = copy_local_images(data_dir, sampled_images)
    else:
        saved_images = download_public_images(sampled_images)

    log(f"Saved annotations to: {OUTPUT_ANNOTATIONS_PATH}")
    log(f"Saved images to: {OUTPUT_IMAGE_DIR}")
    log(f"Requested sample size: {len(sampled_images)}")
    log(f"Saved image files: {saved_images}")


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample_size must be a positive integer.")

    data_dir = args.data_dir.resolve() if args.data_dir else None
    if data_dir is not None and not data_dir.exists():
        raise SystemExit(f"Provided data directory does not exist: {data_dir}")

    if data_dir is None:
        log("Using the public ENA24 source.")
        log(f"Official dataset page: {ENA24_INFO_URL}")
    else:
        log(f"Using local ENA24 directory: {data_dir}")

    annotations, source_mode = load_annotations(data_dir)
    sampled_images = sample_images(
        annotations=annotations,
        data_dir=data_dir,
        source_mode=source_mode,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    save_sample(
        annotations=annotations,
        sampled_images=sampled_images,
        data_dir=data_dir,
        source_mode=source_mode,
    )


if __name__ == "__main__":
    main()
