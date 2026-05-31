import sys
from datetime import datetime
from pathlib import Path

import click
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

# Dzięki temu importy z src działają niezależnie od tego,
# czy odpalasz skrypt z katalogu projektu, czy z katalogu scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.datasets.ena24_baseline_datamodule import ENA24BaselineDataModule
from src.models.baseline.cnn_classifier import SimpleCNN
from src.models.baseline.resnet_classifier import ResNetBinaryClassifier
from src.models.baseline.lightning_sliding_window_detector import LitSlidingWindowCNNDetector


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    # Stary styl: ścieżka liczona tak, jakby skrypt był odpalany z scripts/
    scripts_relative = (PROJECT_ROOT / "scripts" / path).resolve()

    # Nowy styl: ścieżka liczona od katalogu projektu
    project_relative = (PROJECT_ROOT / path).resolve()

    if scripts_relative.exists():
        return scripts_relative

    if project_relative.exists():
        return project_relative

    # Dla plików wyjściowych, np. checkpoints, których jeszcze nie ma.
    if str(path).startswith(".."):
        return scripts_relative

    return project_relative


def normalize_config_paths(config: dict) -> dict:

    if "data" in config and "data_dir" in config["data"]:
        config["data"]["data_dir"] = str(resolve_project_path(config["data"]["data_dir"]))

    if "cnn_training" in config:
        if "checkpoint_path" in config["cnn_training"]:
            config["cnn_training"]["checkpoint_path"] = str(
                resolve_project_path(config["cnn_training"]["checkpoint_path"])
            )

        if "best_checkpoint_path" in config["cnn_training"]:
            config["cnn_training"]["best_checkpoint_path"] = str(
                resolve_project_path(config["cnn_training"]["best_checkpoint_path"])
            )

    return config


def build_cnn(config: dict):
    model_name = config["cnn_training"].get("model", "simple_cnn")

    if model_name == "resnet":
        return ResNetBinaryClassifier(pretrained=True)

    if model_name == "simple_cnn":
        return SimpleCNN()

    raise ValueError(f"Unknown model: {model_name}")


def export_cnn_state_dict_from_lightning_checkpoint(
    lightning_checkpoint_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Lightning zapisuje cały LightningModule jako .ckpt.
    Twój evaluate_baseline.py oczekuje natomiast zwykłego cnn.state_dict() jako .pt.

    Ta funkcja wyciąga z checkpointu Lightninga tylko self.cnn
    i zapisuje go w starym formacie, żeby evaluate_baseline.py nadal działał.
    """
    lightning_checkpoint_path = Path(lightning_checkpoint_path)
    output_path = Path(output_path)

    checkpoint = torch.load(lightning_checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    cnn_state_dict = {
        key.removeprefix("cnn."): value
        for key, value in state_dict.items()
        if key.startswith("cnn.")
    }

    if not cnn_state_dict:
        raise RuntimeError(
            f"No CNN weights found in Lightning checkpoint: {lightning_checkpoint_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cnn_state_dict, output_path)

    print(f"Exported CNN checkpoint: {output_path}")


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default=str(PROJECT_ROOT / "src" / "config" / "baseline_config.json"),
    show_default=True,
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--max_epochs", type=int, default=None)
@click.option("--resume_from_ckpt", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--test", "run_test", is_flag=True, default=False)
def main(
    config_path: str,
    seed: int,
    max_epochs: int | None,
    resume_from_ckpt: str | None,
    run_test: bool,
):
    L.seed_everything(seed)

    config = load_config(config_path)
    config = normalize_config_paths(config)

    run_name = f"baseline_sliding_window_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = PROJECT_ROOT / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Config:", config_path)
    print("Log dir:", log_dir)
    print("Data dir:", config["data"]["data_dir"])

    datamodule = ENA24BaselineDataModule(config=config)

    cnn = build_cnn(config)

    model = LitSlidingWindowCNNDetector(
        cnn=cnn,
        config=config,
    )

    monitor_metric = config.get("lightning", {}).get("monitor", "cnn_val_loss")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(log_dir),
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="baseline-cnn-{epoch:02d}-{" + monitor_metric + ":.4f}",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs or config["cnn_training"]["num_epochs"],
        accelerator="cpu",
        devices=1,
        default_root_dir=str(log_dir),
        callbacks=[checkpoint_callback],
        logger=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_from_ckpt,
    )

    print("Best Lightning checkpoint:", checkpoint_callback.best_model_path)

    # Eksport best checkpoint do starego formatu .pt,
    # żeby scripts/evaluate_baseline.py działał bez przerabiania.
    if checkpoint_callback.best_model_path:
        export_cnn_state_dict_from_lightning_checkpoint(
            lightning_checkpoint_path=checkpoint_callback.best_model_path,
            output_path=config["cnn_training"]["best_checkpoint_path"],
        )

    # Eksport last checkpoint do starego formatu .pt.
    last_ckpt = log_dir / "last.ckpt"
    if last_ckpt.exists():
        export_cnn_state_dict_from_lightning_checkpoint(
            lightning_checkpoint_path=last_ckpt,
            output_path=config["cnn_training"]["checkpoint_path"],
        )

    if run_test:
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path or None,
        )


if __name__ == "__main__":
    main()