"""Check frozen test2 identity/overlap, without model scores or test tuning."""
import gzip
import json
import shutil

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.preprocess_hai_mixed import sha256


def feature_hash(frame, columns):
    # Normalize dtypes so integer/float storage differences cannot hide duplicate rows.
    return pd.util.hash_pandas_object(frame[columns].astype(np.float64), index=False).to_numpy()


def main():
    root = PROJECT_ROOT
    reservation_path = root / "manifests/hai_mixed_v2_holdout.json"
    reservation = json.loads(reservation_path.read_text())
    archive = root / reservation["archive"]
    if sha256(archive) != reservation["archive_sha256"]:
        raise ValueError("Holdout archive differs from its reserved hash")
    destination = root / "data/raw/hai/holdout/test2.csv"
    if not destination.exists():
        with gzip.open(archive, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    # Verify any existing extraction against the reserved archive too.
    import hashlib
    with gzip.open(archive, "rb") as stream:
        expected_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha256(destination) != expected_hash:
        raise ValueError("Extracted holdout does not match reserved archive")
    schema = json.loads((root / "configs/hai_20_07_schema.json").read_text())
    manifest = json.loads((root / "artifacts/hai/preprocessing_hai_mixed_v2/manifest.json").read_text())
    features = schema["feature_order"]
    holdout = pd.read_csv(destination, sep=";")
    if set(holdout) != {"time", *features, "attack", "attack_P1", "attack_P2", "attack_P3"}:
        raise ValueError("Holdout schema differs")
    times = pd.to_datetime(holdout.time, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(seconds=1)).all():
        raise ValueError("Holdout is not a continuous one-second segment")
    if not np.isfinite(holdout[features].to_numpy(dtype=np.float64)).all():
        raise ValueError("Holdout has invalid numeric features")
    hashes = feature_hash(holdout, features)
    comparisons = {}
    for name in ("train1", "train2", "test1"):
        path = root / f"data/raw/hai/{name}.csv"
        if name in manifest["source_audit"] and sha256(path) != manifest["source_audit"][name]["sha256"]:
            raise ValueError(f"Development source changed: {name}")
        frame = pd.read_csv(path, sep=";")
        overlap_times = int(holdout.time.isin(frame.time).sum())
        source_hashes = feature_hash(frame, features)
        candidates = np.flatnonzero(np.isin(hashes, source_hashes))
        # Confirm candidate hash matches using exact floating-point feature vectors.
        exact = 0
        for index in candidates:
            matching = np.flatnonzero(source_hashes == hashes[index])
            target = holdout.iloc[index][features].to_numpy(dtype=np.float64)
            if any(np.array_equal(frame.iloc[j][features].to_numpy(dtype=np.float64), target) for j in matching):
                exact += 1
        comparisons[name] = {
            "shared_timestamps": overlap_times,
            "exact_duplicate_feature_rows_ignoring_time": exact,
            "scope": "full raw source, including excluded rows",
        }
    latest_development = max(pd.Timestamp(info["end"]) for info in manifest["splits"].values())
    passed = all(not row["shared_timestamps"] and not row["exact_duplicate_feature_rows_ignoring_time"]
                 for row in comparisons.values()) and times.iloc[0] > latest_development
    report = {
        "passed": passed, "protocol": "hai_mixed_v2",
        "holdout": str(destination.relative_to(root)), "archive_sha256": reservation["archive_sha256"],
        "extracted_sha256": expected_hash, "rows": len(holdout),
        "start": str(times.iloc[0]), "end": str(times.iloc[-1]),
        "latest_development_timestamp": str(latest_development), "comparisons": comparisons,
        "test_scores_computed": False,
        "limitation": "No overlap/duplicate records found does not prove physical independence of all operating scenarios.",
    }
    path = root / "manifests/hai_mixed_v2_holdout_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise ValueError("Holdout overlap check failed; evaluation must not proceed")


if __name__ == "__main__":
    main()
