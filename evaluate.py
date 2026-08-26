"""Evaluate saved hybrid artifacts and produce optional SHAP explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from autoencoder import NormalAutoencoder
from preprocess import load_csv
from train_pipeline import augment


def recall_at_top_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    selected = np.argsort(scores)[-count:]
    positives = y_true.sum()
    return float(y_true[selected].sum() / positives) if positives else float("nan")


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--data", required=True)
    args.add_argument("--artifacts", default="artifacts")
    args.add_argument("--target", default="TARGET")
    args.add_argument("--output", default=None, help="Optional CSV path for UNIQUE_ID and prediction.")
    args.add_argument("--shap", action="store_true")
    parsed = args.parse_args()
    root = Path(parsed.artifacts)
    raw, y = load_csv(parsed.data, target=parsed.target)
    preprocessor = joblib.load(root / "preprocessor.joblib")
    ae = NormalAutoencoder.load(root / "autoencoder.pt")
    model = joblib.load(root / "xgb_model.joblib")
    x = augment(preprocessor.transform(raw), ae)
    scores = model.predict_proba(x)[:, 1]
    result = {"rows": len(raw), "score_mean": float(scores.mean())}
    if parsed.output:
        if "UNIQUE_ID" not in raw:
            raise ValueError("UNIQUE_ID is required when writing prediction output.")
        output = pd.DataFrame({"UNIQUE_ID": raw["UNIQUE_ID"].to_numpy(), "PREDICTION": scores})
        output.to_csv(parsed.output, index=False)
        result["prediction_output"] = parsed.output
    if y is not None:
        y_np = y.to_numpy()
        result.update({
            "pr_auc_average_precision": float(average_precision_score(y_np, scores)),
            "roc_auc": float(roc_auc_score(y_np, scores)),
            "recall_at_top_1pct": recall_at_top_k(y_np, scores, 0.01),
            "recall_at_top_5pct": recall_at_top_k(y_np, scores, 0.05),
            "recall_at_top_10pct": recall_at_top_k(y_np, scores, 0.10),
        })
    if parsed.shap:
        import shap
        explanation = shap.TreeExplainer(model)(x)
        names = list(preprocessor.feature_names_) + [f"latent_{i}" for i in range(ae.config.latent_dim)] + ["reconstruction_error"]
        importance = np.abs(explanation.values).mean(axis=0)
        result["shap_mean_abs_top20"] = {
            name: float(value)
            for name, value in sorted(
                zip(names, importance), key=lambda item: item[1], reverse=True
            )[:20]
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
