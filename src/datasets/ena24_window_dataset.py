import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.detection.IoU import max_iou_with_true_boxes


class ENA24WindowDataset(Dataset):
    def __init__(
        self,
        samples,
        crop_size=128,
        negative_per_positive=3,
        negative_iou_threshold=0.2,
        seed=42,
    ):
        self.samples = samples
        self.crop_size = crop_size
        self.negative_per_positive = negative_per_positive
        self.negative_iou_threshold = negative_iou_threshold

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.rng = random.Random(seed)
        self.items = self._make_items()

    def _make_items(self):
        items = []

        for sample in self.samples:
            image_path = sample["image_path"]
            width = int(sample["width"])
            height = int(sample["height"])
            true_bboxes = sample["bboxes"]

            # POZYTYWY: prawdziwe boxy zwierząt
            for true_bbox in true_bboxes:
                items.append({
                    "image_path": image_path,
                    "box": true_bbox,
                    "label": 1,
                })

            # NEGATYWY: losowe okna z tła
            number_of_negatives = len(true_bboxes) * self.negative_per_positive

            for _ in range(number_of_negatives):
                negative_box = self._random_background_box(
                    width,
                    height,
                    true_bboxes,
                )

                if negative_box is not None:
                    items.append({
                        "image_path": image_path,
                        "box": negative_box,
                        "label": 0,
                    })

        self.rng.shuffle(items)
        return items

    def _random_background_box(self, image_width, image_height, true_bboxes):
        # próbujemy kilka razy znaleźć okno, które nie nachodzi na zwierzę
        for _ in range(100):
            box_size = self.rng.choice([64, 128, 256])

            if box_size >= image_width or box_size >= image_height:
                continue

            x1 = self.rng.randint(0, image_width - box_size)
            y1 = self.rng.randint(0, image_height - box_size)
            x2 = x1 + box_size
            y2 = y1 + box_size

            bbox = [x1, y1, x2, y2]

            if max_iou_with_true_boxes(bbox, true_bboxes) < self.negative_iou_threshold:
                return bbox

        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        image_path = item["image_path"]
        box = item["box"]
        label = item["label"]

        x1, y1, x2, y2 = box

        image = Image.open(image_path).convert("RGB")
        image = image.crop((x1, y1, x2, y2))

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

