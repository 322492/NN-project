from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    scripts_relative = (PROJECT_ROOT / "scripts" / path).resolve()
    project_relative = (PROJECT_ROOT / path).resolve()

    if scripts_relative.exists():
        return scripts_relative

    if project_relative.exists():
        return project_relative

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

    if "outputs" in config and "result_image_path" in config["outputs"]:
        config["outputs"]["result_image_path"] = str(
            resolve_project_path(config["outputs"]["result_image_path"])
        )

    return config
