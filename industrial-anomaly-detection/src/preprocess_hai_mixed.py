"""Prepare isolated HAI mixed-target artifacts; never fit on validation/test."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_schema(schema):
    numeric = schema["numeric_columns"]
    binary = schema["binary_columns"]
    features = schema["feature_order"]
    if (len(numeric) != 55 or len(binary) != 4
            or len(set(numeric + binary)) != 59
            or len(features) != 59 or set(features) != set(numeric + binary)
            or set(features) & set(schema["excluded_columns"])):
        raise ValueError("Invalid HAI 20.07 channel schema")


def validate_frame(frame, schema):
    expected = ["time", *schema["feature_order"],
                "attack", "attack_P1", "attack_P2", "attack_P3"]
    if frame.empty or list(frame.columns) != expected:
        raise ValueError("Empty data or unexpected column order/schema")
    times = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("Expected continuous 1-second timestamps")
    for name in ("attack", "attack_P1", "attack_P2", "attack_P3"):
        if not frame[name].eq(0).all():
            raise ValueError("Preparation requires normal-only train/validation")
    for name in schema["feature_order"]:
        if not np.isfinite(pd.to_numeric(frame[name], errors="raise")).all():
            raise ValueError(f"Missing/non-finite values in {name}")
    if not frame[schema["binary_columns"]].isin([0, 1]).all().all():
        raise ValueError("Binary states must be 0/1")
    return times


def transform_frame(frame, bundle):
    """Transform with a fitted bundle, preserving explicit grouped input order."""
    numeric = bundle["scaler"].transform(frame[bundle["numeric_columns"]])
    binary = frame[bundle["binary_columns"]].to_numpy()
    if not np.isfinite(numeric).all() or not np.isin(binary, [0, 1]).all():
        raise ValueError("Invalid transformed values")
    numeric = numeric.astype(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric values overflow float32")
    return numeric, binary.astype(np.int64)


def prepare(train_path, validation_path, schema_path, output_dir, artifact_dir):
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validate_schema(schema)
    train = pd.read_csv(train_path, sep=";")
    validation = pd.read_csv(validation_path, sep=";")
    train_times = validate_frame(train, schema)
    validation_times = validate_frame(validation, schema)
    if train_times.iloc[-1] >= validation_times.iloc[0]:
        raise ValueError("Train and validation overlap or are out of order")

    scaler = StandardScaler().fit(train[schema["numeric_columns"]])
    bundle = {
        "format_version": 1,
        "dataset": "HAI 20.07",
        "scaler": scaler,
        "numeric_columns": schema["numeric_columns"],
        "binary_columns": schema["binary_columns"],
        "input_columns": schema["numeric_columns"] + schema["binary_columns"],
        "binary_classes": [0, 1],
        "schema_sha256": sha256(schema_path),
        "source_sha256": {"train": sha256(train_path), "validation": sha256(validation_path)},
    }
    transformed = {
        name: transform_frame(frame, bundle)
        for name, frame in (("train", train), ("validation", validation))
    }
    output_dir, artifact_dir = Path(output_dir), Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_dir / "preprocessing.joblib")

    metadata = {key: value for key, value in bundle.items() if key != "scaler"}
    metadata["splits"] = {}
    for name, frame in (("train", train), ("validation", validation)):
        numeric, binary = transformed[name]
        np.savez_compressed(
            output_dir / f"{name}.npz",
            numeric=numeric, binary=binary,
            timestamps=frame["time"].to_numpy(dtype=str),
            labels=frame["attack"].to_numpy(dtype=np.int64),
            numeric_columns=np.array(bundle["numeric_columns"], dtype=str),
            binary_columns=np.array(bundle["binary_columns"], dtype=str),
            schema_sha256=np.array(bundle["schema_sha256"]),
        )
        metadata["splits"][name] = {
            "rows": len(frame), "start": frame["time"].iloc[0],
            "end": frame["time"].iloc[-1],
        }
        print(f"{name}: numeric={numeric.shape}, binary={binary.shape}")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return bundle


def main():
    prepare(
        PROJECT_ROOT / "data/processed/hai/train.csv",
        PROJECT_ROOT / "data/processed/hai/validation.csv",
        PROJECT_ROOT / "configs/hai_20_07_schema.json",
        PROJECT_ROOT / "data/processed/hai_mixed",
        PROJECT_ROOT / "artifacts/hai/preprocessing_mixed",
    )
    print("Mixed preprocessing completed (test data not used)")


if __name__ == "__main__":
    main()
