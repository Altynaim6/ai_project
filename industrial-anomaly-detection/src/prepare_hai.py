import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/hai/train1.csv"
OUTPUT = ROOT / "data/processed/hai"
MANIFEST = ROOT / "manifests/hai_split.json"


def main():
    frame = pd.read_csv(SOURCE, sep=";")
    frame.columns = frame.columns.str.strip()

    times = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    if frame.empty or not frame["attack"].eq(0).all():
        raise ValueError("Expected non-empty, normal-only training data")

    if not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("Expected continuous timestamps with 1-second step")

    boundary = int(len(frame) * 0.8)
    train = frame.iloc[:boundary].copy()
    validation = frame.iloc[boundary:].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем исходные значения без нормализации.
    train.to_csv(OUTPUT / "train.csv", sep=";", index=False)
    validation.to_csv(OUTPUT / "validation.csv", sep=";", index=False)

    manifest = {
        "dataset": "HAI 20.07",
        "purpose": "temporary normal-only forecasting experiment",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.file_digest(
            SOURCE.open("rb"), "sha256"
        ).hexdigest(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_source_indices": [0, boundary],
        "validation_source_indices": [boundary, len(frame)],
        "index_convention": "zero-based, end exclusive",
        "train_start": train["time"].iloc[0],
        "train_end": train["time"].iloc[-1],
        "validation_start": validation["time"].iloc[0],
        "validation_end": validation["time"].iloc[-1],
        "test_source": "data/raw/hai/test1.csv",
        "window_rule": "Build windows separately within each split",
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Train rows:", len(train))
    print("Validation rows:", len(validation))
    print("Saved to:", OUTPUT)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()