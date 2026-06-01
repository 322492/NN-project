from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.data_splits import prepare_data_splits
from src.datasets.ena24_window_dataset import ENA24WindowDataset
from src.models.baseline.cnn_classifier import train_one_epoch, evaluate
from src.utils.wandb_utils import log_model_artifact


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

    train_window_loader = DataLoader(
        train_window_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_window_loader = DataLoader(
        val_window_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("train images:", len(train_samples))
    print("val images:", len(val_samples))
    print("train CNN crops:", len(train_window_dataset))
    print("val CNN crops:", len(val_window_dataset))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, cnn.parameters()),
        lr=config["cnn_training"]["learning_rate"],
    )

    num_epochs = config["cnn_training"]["num_epochs"]
    threshold = config["cnn_training"]["threshold"]

    last_checkpoint_path = Path(
        config["cnn_training"].get(
            "checkpoint_path",
            "../checkpoints/baseline_resnet.pt",
        )
    )

    best_checkpoint_path = Path(
        config["cnn_training"].get(
            "best_checkpoint_path",
            str(last_checkpoint_path.with_name(last_checkpoint_path.stem + "_best" + last_checkpoint_path.suffix)),
        )
    )

    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_loss_acc = 0.0
    best_val_loss_epoch = 0

    best_val_acc = 0.0
    best_val_acc_epoch = 0

    final_val_loss = None
    final_val_acc = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            cnn,
            train_window_loader,
            optimizer,
            criterion,
            device,
            threshold=threshold,
        )

        val_loss, val_acc = evaluate(
            cnn,
            val_window_loader,
            criterion,
            device,
            threshold=threshold,
        )

        final_val_loss = val_loss
        final_val_acc = val_acc

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

    print(f"Saved last checkpoint: {last_checkpoint_path}")
    print(f"Saved best checkpoint: {best_checkpoint_path}")

    if run is not None:
        run.summary["cnn_best_val_loss"] = best_val_loss
        run.summary["cnn_best_val_loss_acc"] = best_val_loss_acc
        run.summary["cnn_best_val_loss_epoch"] = best_val_loss_epoch

        run.summary["cnn_best_val_acc"] = best_val_acc
        run.summary["cnn_best_val_acc_epoch"] = best_val_acc_epoch

        run.summary["cnn_final_val_loss"] = final_val_loss
        run.summary["cnn_final_val_acc"] = final_val_acc

        log_model_artifact(
            run=run,
            checkpoint_path=best_checkpoint_path,
            artifact_name="baseline_resnet_best",
        )
