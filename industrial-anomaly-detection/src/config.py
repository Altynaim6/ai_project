import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    config_path = PROJECT_ROOT / "configs" / "base.json"

    with config_path.open(encoding="utf-8-sig") as file:
        config = json.load(file)

    return config


def prepare_directories(config):
    directories = [
        config["dataset"]["raw_directory"],
        config["dataset"]["processed_directory"],
        config["dataset"]["manifest_directory"],
        config["output"]["artifacts_directory"],
        config["output"]["results_directory"],
    ]

    for directory in directories:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)