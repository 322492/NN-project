import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


def xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox
    return [x1, y1, x1 + w, y1 + h]


class ENA24DetectionDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_path = self.data_dir / "annotations.json"
        self.transform = transform

        self.samples = self._load_samples()

        if self.transform is None:
            self.transform = transforms.ToTensor()

    def _load_samples(self):
        with self.annotations_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        images = data["images"]
        annotations = data["annotations"]

        boxes_by_image_id = {}

        for annotation in annotations:
            image_id = annotation["image_id"]
            bbox_xywh = annotation["bbox"]
            bbox_xyxy = xywh_to_xyxy(bbox_xywh)

            if image_id not in boxes_by_image_id:
                boxes_by_image_id[image_id] = []

            boxes_by_image_id[image_id].append(bbox_xyxy)

        samples = []

        for image in images:
            image_id = image["id"]
            file_name = image["file_name"]
            width = image["width"]
            height = image["height"]

            image_path = self.images_dir / Path(file_name).name

            sample = {
                "image_id": image_id,
                "file_name": Path(file_name).name,
                "image_path": image_path,
                "width": width,
                "height": height,
                "bboxes": boxes_by_image_id.get(image_id, []),
            }

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = {
            "image_id": sample["image_id"],
            "image_path": str(sample["image_path"]),
            "width": sample["width"],
            "height": sample["height"],
            "bboxes": torch.tensor(sample["bboxes"], dtype=torch.float32),
        }

        return image, target
