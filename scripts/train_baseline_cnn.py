from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.ena24_dataset import ENA24DetectionDataset
from src.datasets.ena24_window_dataset import ENA24WindowDataset
from src.models.baseline.cnn_classifier import train_one_epoch, evaluate
from src.utils.wandb_utils import log_model_artifact


def prepare_data_splits(data_dir="../data/ena24_sample", train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)

    full_dataset = ENA24DetectionDataset(
        data_dir=data_dir,
    )

    train_size = int(train_ratio * len(full_dataset))
    val_size = int(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_detection_dataset, val_detection_dataset, test_detection_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_samples = [full_dataset.samples[i] for i in train_detection_dataset.indices]
    val_samples = [full_dataset.samples[i] for i in val_detection_dataset.indices]
    test_samples = [full_dataset.samples[i] for i in test_detection_dataset.indices]

    return full_dataset, train_samples, val_samples, test_samples


def train(cnn, criterion, device, train_samples, val_samples, config, run=None):
    train_window_dataset = ENA24WindowDataset(
        train_samples,
        crop_size=config["cnn_dataset"]["crop_size"],
        negative_per_positive=config["cnn_dataset"]["negative_per_positive"],
        negative_iou_threshold=config["cnn_dataset"]["negative_iou_threshold"],
    )

    val_window_dataset = ENA24WindowDataset(
        val_samples,
        crop_size=config["cnn_dataset"]["crop_size"],
        negative_per_positive=config["cnn_dataset"]["negative_per_positive"],
        negative_iou_threshold=config["cnn_dataset"]["negative_iou_threshold"],
    )

    batch_size = config["cnn_training"].get("batch_size", 32)

    train_window_loader = DataLoader(train_window_dataset, batch_size=batch_size, shuffle=True)
    val_window_loader = DataLoader(val_window_dataset, batch_size=batch_size, shuffle=False)

    print("train images:", len(train_samples))
    print("val images:", len(val_samples))
    print("train CNN crops:", len(train_window_dataset))
    print("val CNN crops:", len(val_window_dataset))


    optimizer = torch.optim.Adam(cnn.parameters(), lr=config["cnn_training"]["learning_rate"])
    num_epochs = config["cnn_training"]["num_epochs"]

    checkpoint_dir = Path("../checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_checkpoint_path = checkpoint_dir / "baseline_cnn_best.pt"
    last_checkpoint_path = checkpoint_dir / "baseline_cnn.pt"

    best_val_loss = float("inf")
    best_val_loss_acc = 0.0
    best_val_loss_epoch = 0

    best_val_acc = 0.0
    best_val_acc_epoch = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            cnn,
            train_window_loader,
            optimizer,
            criterion,
            device,
            threshold=config["cnn_training"]["threshold"]
        )

        val_loss, val_acc = evaluate(
            cnn,
            val_window_loader,
            criterion,
            device,
            threshold=config["cnn_training"]["threshold"]
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_acc = val_acc
            best_val_loss_epoch = epoch + 1

            torch.save(cnn.state_dict(), best_checkpoint_path)

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"train acc: {train_acc:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val acc: {val_acc:.4f}"
            )

        if run is not None:
            run.log({
                "epoch": epoch + 1,
                "cnn_train_loss": train_loss,
                "cnn_train_acc": train_acc,
                "cnn_val_loss": val_loss,
                "cnn_val_acc": val_acc,
            })

    torch.save(cnn.state_dict(), last_checkpoint_path)
    print("Saved last checkpoint:", last_checkpoint_path)
    print("Best checkpoint:", best_checkpoint_path)

    if run is not None:
        run.summary["cnn_best_val_loss"] = best_val_loss
        run.summary["cnn_best_val_loss_acc"] = best_val_loss_acc
        run.summary["cnn_best_val_loss_epoch"] = best_val_loss_epoch

        run.summary["cnn_best_val_acc"] = best_val_acc
        run.summary["cnn_best_val_acc_epoch"] = best_val_acc_epoch

        run.summary["cnn_final_val_loss"] = val_loss
        run.summary["cnn_final_val_acc"] = val_acc

        log_model_artifact(
            run=run,
            checkpoint_path=best_checkpoint_path,
            artifact_name="baseline_cnn_best"
        )