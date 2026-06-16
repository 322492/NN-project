import lightning as L
from torch.utils.data import DataLoader

from src.datasets.data_splits import prepare_data_splits_from_data_config
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

        (
            self.full_dataset,
            self.train_samples,
            self.val_samples,
            self.test_samples,
        ) = prepare_data_splits_from_data_config(
            config["data"],
            data_dir=config["data"]["data_dir"],
            seed=config["seed"],
        )

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
        # Cropped windows + full images (two val dataloaders)

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
        # Full images for detection eval
        return DataLoader(
            self.test_samples,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=detection_collate_fn,
        )