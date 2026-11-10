"""Audit train2 and prepare versioned normal splits; reserve test2 unopened."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.preprocess_hai_mixed import sha256, transform_frame, validate_schema
from src.range_floor_scaler import RangeFloorScaler


LABELS = ["attack", "attack_P1", "attack_P2", "attack_P3"]


def engineering_ranges():
    # HAI manual v4.0, printed pp.12-15. These are scale references, not clipping limits.
    ranges = {}
    def add(names, low, high):
        for name in names.split():
            ranges[name] = [low, high]
    add("P1_B2004 P1_B2016 P1_PIT01 P1_PIT02", 0, 10)
    add("P1_B3004 P1_LIT01", 0, 720)
    add("P1_B3005 P1_B400B P1_FT01 P1_FT02 P1_FT03", 0, 2500)
    add("P1_B4002 P1_B4005", 0, 100)
    add("P1_B4022", 0, 40)
    for valve in ("FCV01", "FCV02", "FCV03", "LCV01", "PCV01", "PCV02"):
        add(f"P1_{valve}D P1_{valve}Z", 0, 100)
    add("P1_FT01Z P1_FT02Z P1_FT03Z", 0, 3190)
    add("P1_TIT01 P1_TIT02", -50, 150)
    add("P2_24Vdc", 0, 30)
    add("P2_SD01 P2_SIT01", 0, 3200)
    add("P2_VXT02 P2_VXT03 P2_VYT02 P2_VYT03", -10, 10)
    add("P2_VT01e", 11, 12)
    add("P3_LCP01D P3_LCV01D P4_ST_PT01 P4_ST_TT01", 0, 27648)
    add("P3_LH P3_LL", 0, 70)
    add("P3_LT01", 0, 90)
    add("P4_HT_FD P4_ST_FD", -.02, .02)
    add("P4_HT_LD P4_HT_PO P4_HT_PS", 0, 100)
    add("P4_LD P4_ST_LD P4_ST_PO P4_ST_PS", 0, 500)
    return ranges


def attack_exclusion_mask(frame, guard_seconds=300):
    """Offline normal-data cleaning: exclude any attack flag and a fixed guard."""
    if guard_seconds < 0:
        raise ValueError("guard_seconds must be non-negative")
    attack = frame[LABELS].ne(0).any(axis=1)
    # Input already validated as one-second continuous data.
    exclude = attack.rolling(2 * guard_seconds + 1, center=True, min_periods=1).max().astype(bool)
    return attack.to_numpy(), exclude.to_numpy()


def audit_source(path, schema):
    frame = pd.read_csv(path, sep=";")
    expected = ["time", *schema["feature_order"], *LABELS]
    if list(frame.columns) != expected or frame.empty:
        raise ValueError(f"Unexpected schema: {path}")
    times = pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError(f"Not a continuous one-second source: {path}")
    values = frame[schema["feature_order"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing/non-finite features: {path}")
    if not frame[LABELS + schema["binary_columns"]].isin([0, 1]).all().all():
        raise ValueError(f"Invalid state/attack labels: {path}")
    return frame


def segment_count(frame):
    times = pd.to_datetime(frame["time"])
    return int(times.diff().ne(pd.Timedelta(seconds=1)).sum())


def prepare_v2(root=PROJECT_ROOT, version="hai_mixed_v2"):
    root = Path(root)
    schema_path = root / "configs/hai_20_07_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_schema(schema)
    output = root / "data/processed" / version
    artifacts = root / "artifacts/hai" / f"preprocessing_{version}"
    report_dir = root / "reports" / version
    plan_path = root / "manifests" / f"{version}_split.json"
    if any(p.exists() for p in (output, artifacts, report_dir, plan_path)):
        raise FileExistsError("Version already exists; choose a new version rather than overwriting")
    plan = {
        "version": version, "dataset": "HAI 20.07",
        "development_sources": ["train1.csv", "train2.csv"],
        "previous_test1_status": "Already inspected; diagnostic only, not a fresh test",
        "fresh_test": {"file": "data/raw/hai/test2.csv", "status": "reserved_uninspected",
                       "source_url": "https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/test2.csv.gz",
                       "opened_by_preprocessing": False,
                       "independence_status": "Must be verified before future one-time evaluation"},
        "rule": "All normal train1 + first 70% train2 for training; next 15% train2 for model selection; last 15% train2 for threshold calibration",
        "normal_filter": "All four attack flags must be zero; remove +/-300 seconds around every attack flag, within source",
        "window_rule": "Never cross source, split or excluded-interval boundaries",
        "scale_rule": "max(train std, 0.01 * documented engineering range width); train mean centering; no clipping",
        "loss": "mean numeric MSE + mean two-class binary CE; unchanged weights 1:1",
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    frames = {name: audit_source(root / "data/raw/hai" / f"{name}.csv", schema)
              for name in ("train1", "train2")}
    if pd.Timestamp(frames["train1"].time.iloc[-1]) >= pd.Timestamp(frames["train2"].time.iloc[0]):
        raise ValueError("Development sources overlap or are out of order")
    n = len(frames["train2"])
    cut1, cut2 = int(n * .70), int(n * .85)
    clean = {}
    source_audit = {}
    for name, frame in frames.items():
        attack, excluded = attack_exclusion_mask(frame)
        clean[name] = ~excluded
        starts = np.flatnonzero(attack & ~np.r_[False, attack[:-1]])
        ends = np.flatnonzero(attack & ~np.r_[attack[1:], False])
        source_audit[name] = {
            "rows": len(frame), "start": frame.time.iloc[0], "end": frame.time.iloc[-1],
            "sha256": sha256(root / "data/raw/hai" / f"{name}.csv"),
            "attack_flagged_rows": int(attack.sum()), "excluded_with_guard": int(excluded.sum()),
            "attack_intervals": [{"start": frame.time.iloc[a], "end": frame.time.iloc[b],
                                  "rows": int(b-a+1)} for a, b in zip(starts, ends)],
        }
    train2 = frames["train2"]
    splits = {
        "train": pd.concat([
            frames["train1"].loc[clean["train1"]],
            train2.iloc[:cut1].loc[clean["train2"][:cut1]],
        ], ignore_index=True),
        "validation": train2.iloc[cut1:cut2].loc[clean["train2"][cut1:cut2]].reset_index(drop=True),
        "calibration": train2.iloc[cut2:].loc[clean["train2"][cut2:]].reset_index(drop=True),
    }
    if any(len(frame) <= 30 for frame in splits.values()):
        raise ValueError("A cleaned split is too short")
    ranges = engineering_ranges()
    if set(ranges) != set(schema["numeric_columns"]):
        raise ValueError("Engineering ranges do not cover exactly the numeric schema")
    scaler = RangeFloorScaler(ranges, floor_fraction=.01).fit(splits["train"][schema["numeric_columns"]])
    bundle = {
        "format_version": 2, "dataset": "HAI 20.07", "preprocessing_version": version,
        "scaler": scaler, "numeric_columns": schema["numeric_columns"],
        "binary_columns": schema["binary_columns"],
        "input_columns": schema["numeric_columns"] + schema["binary_columns"],
        "binary_classes": [0, 1], "schema_sha256": sha256(schema_path),
        "engineering_ranges": ranges, "scale_floor_fraction": .01,
        "source_sha256": {name: info["sha256"] for name, info in source_audit.items()},
    }
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    joblib.dump(bundle, artifacts / "preprocessing.joblib")
    bundle_hash = sha256(artifacts / "preprocessing.joblib")
    metadata = {key: value for key, value in bundle.items() if key != "scaler"}
    metadata.update(protocol=plan, source_audit=source_audit, splits={}, preprocessing_sha256=bundle_hash)
    for name, frame in splits.items():
        numeric, binary = transform_frame(frame, bundle)
        np.savez_compressed(
            output / f"{name}.npz", numeric=numeric, binary=binary,
            timestamps=frame.time.to_numpy(dtype=str), labels=frame.attack.to_numpy(dtype=np.int64),
            numeric_columns=np.array(schema["numeric_columns"]),
            binary_columns=np.array(schema["binary_columns"]),
            schema_sha256=np.array(bundle["schema_sha256"]),
            preprocessing_sha256=np.array(bundle_hash),
        )
        metadata["splits"][name] = {
            "rows": len(frame), "start": frame.time.iloc[0], "end": frame.time.iloc[-1],
            "segments": segment_count(frame),
        }
        print(name, metadata["splits"][name])
    audit = pd.DataFrame({
        "channel": schema["numeric_columns"], "train_mean": scaler.mean_,
        "train_std": np.sqrt(scaler.var_), "engineering_floor": scaler.scale_floor_,
        "effective_scale": scaler.scale_, "floor_applied": scaler.floor_applied_,
    })
    audit.to_csv(report_dir / "numeric_scaling.csv", index=False)
    (artifacts / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_dir / "source_audit.json").write_text(json.dumps(source_audit, indent=2), encoding="utf-8")
    print("Scale floor applied:", int(scaler.floor_applied_.sum()), "of", len(scaler.scale_))
    print("Fresh test2 reserved; neither test1 nor test2 was read by this preparation")
    return metadata


if __name__ == "__main__":
    prepare_v2()
