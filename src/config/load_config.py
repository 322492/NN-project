import json
from pathlib import Path


def load_config(config_path):
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return config