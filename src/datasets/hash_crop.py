"""Crop images before perceptual hashing (removes timestamp/camera overlay)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.datasets.coco_io import load_json, save_json
from src.datasets.pipeline_metadata import PIPELINE_VERSION, utc_now_iso

DEFAULT_HASH_CONFIG_FILENAME = "hash_config.json"


@dataclass(frozen=True)
class HashCropConfig:
    top_frac: float = 0.06
    bottom_frac: float = 0.10
    left_frac: float = 0.0
    right_frac: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("top_frac", self.top_frac),
            ("bottom_frac", self.bottom_frac),
            ("left_frac", self.left_frac),
            ("right_frac", self.right_frac),
        ):
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0), got: {value}")

        if self.top_frac + self.bottom_frac >= 1.0:
            raise ValueError("top_frac + bottom_frac must be less than 1.0")

        if self.left_frac + self.right_frac >= 1.0:
            raise ValueError("left_frac + right_frac must be less than 1.0")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HashCropConfig:
        return cls(
            top_frac=float(data.get("top_frac", 0.06)),
            bottom_frac=float(data.get("bottom_frac", 0.10)),
            left_frac=float(data.get("left_frac", 0.0)),
            right_frac=float(data.get("right_frac", 0.0)),
        )


def crop_for_hashing(
    image: Image.Image,
    top_frac: float = 0.06,
    bottom_frac: float = 0.10,
    left_frac: float = 0.0,
    right_frac: float = 0.0,
) -> Image.Image:
    """Crop top/bottom (and optional side) overlay bands. Fractions are relative to input size."""
    config = HashCropConfig(
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        left_frac=left_frac,
        right_frac=right_frac,
    )
    config.validate()

    width, height = image.size

    left = int(round(width * left_frac))
    right = int(round(width * (1.0 - right_frac)))
    top = int(round(height * top_frac))
    bottom = int(round(height * (1.0 - bottom_frac)))

    if right <= left or bottom <= top:
        raise ValueError("Invalid crop box - resulting image would have zero size.")

    return image.crop((left, top, right, bottom))


def crop_for_hashing_with_config(
    image: Image.Image,
    crop_config: HashCropConfig | dict[str, Any] | None = None,
) -> Image.Image:
    if crop_config is None:
        crop_config = HashCropConfig()
    elif isinstance(crop_config, dict):
        crop_config = HashCropConfig.from_dict(crop_config)

    crop_config.validate()
    return crop_for_hashing(
        image,
        top_frac=crop_config.top_frac,
        bottom_frac=crop_config.bottom_frac,
        left_frac=crop_config.left_frac,
        right_frac=crop_config.right_frac,
    )


def save_hash_config(
    config: HashCropConfig,
    path: Path | str,
    *,
    hash_backend: str = "imagehash",
) -> None:
    config.validate()
    save_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "hash_backend": hash_backend,
            "created_at": utc_now_iso(),
            **config.to_dict(),
        },
        path,
    )


def load_hash_config(path: Path | str) -> HashCropConfig:
    payload = load_json(path)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid hash_config format: {path}")

    crop_fields = {
        key: payload[key]
        for key in ("top_frac", "bottom_frac", "left_frac", "right_frac")
        if key in payload
    }
    return HashCropConfig.from_dict(crop_fields or payload)


def hash_config_from_fractions(
    top_frac: float = 0.06,
    bottom_frac: float = 0.10,
    left_frac: float = 0.0,
    right_frac: float = 0.0,
) -> HashCropConfig:
    config = HashCropConfig(
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        left_frac=left_frac,
        right_frac=right_frac,
    )
    config.validate()
    return config
