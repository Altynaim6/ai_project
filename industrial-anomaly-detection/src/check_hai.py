import argparse
from pathlib import Path

import pandas as pd


def check_file(path):
    frame = pd.read_csv(path, sep=";")
    frame.columns = frame.columns.str.strip()

    label_columns = ["attack", "attack_P1", "attack_P2", "attack_P3"]
    required = ["time", *label_columns]

    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    timestamps = pd.to_datetime(
        frame["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    feature_columns = [
        name for name in frame.columns if name not in required
    ]
    features = frame[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    print("File:", path.name)
    print("Rows:", len(frame))
    print("Features:", len(feature_columns))
    print("Invalid timestamps:", timestamps.isna().sum())
    print("Duplicate timestamps:", timestamps.dropna().duplicated().sum())
    print("Time ordered:", timestamps.is_monotonic_increasing)

    differences = timestamps.diff().iloc[1:]
    print("Intervals other than 1 second:",
          differences.ne(pd.Timedelta(seconds=1)).sum())

    print("Missing feature values:", frame[feature_columns].isna().sum().sum())
    print("Non-numeric feature values:",
          (features.isna() & frame[feature_columns].notna()).sum().sum())
    print("Infinite feature values:",
          features.isin([float("inf"), float("-inf")]).sum().sum())

    print("\nAttack label counts:")
    print(frame["attack"].value_counts(dropna=False).to_string())

    print("\nInvalid labels per column:")
    for name in label_columns:
        invalid = ~frame[name].isin([0, 1])
        print(f"{name}: {invalid.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    check_file(args.path)