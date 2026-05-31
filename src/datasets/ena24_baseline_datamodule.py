import lightning as L
import torch
from torch.utils.data import DataLoader

from src.datasets.ena24_dataset import ENA24DetectionDataset
from src.datasets.ena24_window_dataset import ENA24WindowDataset


def detection_collate_fn(batch):
    #zwracamy poprostu liste sampli

    return batch


class ENA24BaselineDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.full_dataset = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        self.train_window_dataset = None
        self.val_window_dataset = None

    def setup(self, stage=None):
        config = self.config
        seed = config["seed"]

        torch.manual_seed(seed)

        self.full_dataset = ENA24DetectionDataset(
            data_dir=config["data"]["data_dir"],
        )

        generator = torch.Generator().manual_seed(seed)

        indices = torch.randperm(
            len(self.full_dataset),
            generator=generator,
        ).tolist()

        size = config["data"].get("size")
        if size is not None:
            size = min(size, len(indices))
            indices = indices[:size]

        train_ratio = config["data"]["train_ratio"]
        val_ratio = config["data"]["val_ratio"]

        train_size = int(train_ratio * len(indices))
        val_size = int(val_ratio * len(indices))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_samples = [self.full_dataset.samples[i] for i in train_indices]
        self.val_samples = [self.full_dataset.samples[i] for i in val_indices]
        self.test_samples = [self.full_dataset.samples[i] for i in test_indices]

        self.train_window_dataset = ENA24WindowDataset(
            self.train_samples,
            crop_size=config["cnn_dataset"]["crop_size"],
            negative_per_positive=config["cnn_dataset"]["negative_per_positive"],
            negative_iou_threshold=config["cnn_dataset"]["negative_iou_threshold"],
        )

        self.val_window_dataset = ENA24WindowDataset(
            self.val_samples,
            crop_size=config["cnn_dataset"]["crop_size"],
            negative_per_positive=config["cnn_dataset"]["negative_per_positive"],
            negative_iou_threshold=config["cnn_dataset"]["negative_iou_threshold"],
        )

        print("train images:", len(self.train_samples))
        print("val images:", len(self.val_samples))
        print("test images:", len(self.test_samples))
        print("train CNN crops:", len(self.train_window_dataset))
        print("val CNN crops:", len(self.val_window_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_window_dataset,
            batch_size=self.config["cnn_training"].get("batch_size", 32),
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        #cropy i prłnr obrazy (dwa dataloadery)

        crop_val_loader = DataLoader(
            self.val_window_dataset,
            batch_size=self.config["cnn_training"].get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )

        detector_val_loader = DataLoader(
            self.val_samples,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=detection_collate_fn,
        )

        return [crop_val_loader, detector_val_loader]

    def test_dataloader(self):
        # pełne obrazy
        return DataLoader(
            self.test_samples,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=detection_collate_fn,
        )