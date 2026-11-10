"""Post-evaluation diagnostic only; preserve RF models and thresholds unchanged."""
import json
import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.baseline import make_features
from src.run_random_forest_hai import transformed


def main():
    artifacts = PROJECT_ROOT / "artifacts/hai/random_forest/hai_rf_classifier_v1"
    results = PROJECT_ROOT / "results/hai/random_forest/hai_rf_classifier_v1"
    bundle = joblib.load(artifacts / "preprocessing.joblib")
    source = pd.read_csv(PROJECT_ROOT / "data/raw/hai/train2.csv", sep=";")
    manifest = pd.read_csv(artifacts / "training_targets.csv")
    attacks = manifest.loc[manifest.label.eq(1)]
    if not attacks.source.eq("train2").all():
        raise ValueError("Unexpected attack source")
    features = make_features(transformed(source, bundle), attacks.target_row_index.to_numpy(dtype=np.int64), 30)
    diagnostics = []
    for seed in (42, 43, 44):
        folder = artifacts / f"seed{seed}"
        model = joblib.load(folder / "model.joblib")["baseline"]
        threshold = json.loads((folder / "threshold.json").read_text())["threshold"]
        scores = model.score_samples(features)
        test = pd.read_csv(results / f"seed{seed}/test_predictions.csv")
        diagnostics.append({"seed": seed, "training_attack_targets": len(scores),
                            "training_attack_recall": float(np.mean(scores > threshold)),
                            "training_attack_score_median": float(np.median(scores)),
                            "test_score_max": float(test.score.max()), "threshold": threshold})
    (results / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
