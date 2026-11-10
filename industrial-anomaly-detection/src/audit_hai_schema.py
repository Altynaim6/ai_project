"""Build the HAI 20.07 channel schema using only the existing train split."""
import json

import pandas as pd

from src.config import PROJECT_ROOT


BINARY = {
    "P2_Auto": "Auto start button (manual: P2_AutoGo)",
    "P2_Emgy": "Emergency button (manual: P2_Emerg)",
    "P2_On": "Turbine DCS on/off switch (manual: P2_OnOff)",
    "P2_TripEx": "Trip emergency exit button",
}
EXCLUDED = ["time", "attack", "attack_P1", "attack_P2", "attack_P3"]


def main():
    frame = pd.read_csv(
        PROJECT_ROOT / "data/processed/hai/train.csv", sep=";"
    )
    features = [name for name in frame if name not in EXCLUDED]
    if len(features) != 59 or not set(BINARY).issubset(features):
        raise ValueError("Expected HAI 20.07 schema with 59 channels")

    records = []
    for name in features:
        series = pd.to_numeric(frame[name], errors="raise")
        binary = name in BINARY
        if binary and not series.isin([0, 1]).all():
            raise ValueError(f"Non-binary values in {name}")
        records.append({
            "name": name,
            "type": "binary" if binary else "numeric",
            "train_unique_count": int(series.nunique()),
            "train_min": float(series.min()),
            "train_max": float(series.max()),
            "constant_on_train": bool(series.nunique() == 1),
            "description": BINARY.get(name, "See manual HAI DATA POINTS"),
        })

    schema = {
        "dataset": "HAI 20.07",
        "source": "HAI technical manual v4.0, May 2023, printed pp.12-15",
        "feature_order": features,
        "numeric_columns": [c for c in features if c not in BINARY],
        "binary_columns": [c for c in features if c in BINARY],
        "excluded_columns": EXCLUDED,
        "binary_classes": [0, 1],
        "alias_note": "P2_VT01e is provisionally mapped to P2_VT01 (phase lag); exact suffix is not explained in the manual table.",
        "loss": "mean numeric standardized MSE + mean binary two-class CE",
        "channels": records,
    }
    path = PROJECT_ROOT / "configs/hai_20_07_schema.json"
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print("Numeric channels:", len(schema["numeric_columns"]))
    print("Binary channels:", schema["binary_columns"])
    print("Train constants:", [r["name"] for r in records if r["constant_on_train"]])
    print("Saved:", path)


if __name__ == "__main__":
    main()
