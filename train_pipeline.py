"""Train the temporal hybrid PyTorch Autoencoder + XGBoost pipeline.

Example:
python train_pipeline.py --train HACKATHON_TRAINING_DATA.CSV --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from autoencoder import AutoencoderConfig, NormalAutoencoder
from preprocess import FeaturePreprocessor, load_csv, split_by_time


def augment(x: np.ndarray, ae: NormalAutoencoder) -> np.ndarray:
    latent, reconstruction_error = ae.transform(x)
    return np.hstack([x, latent, reconstruction_error]).astype(np.float32)


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    raw, y = load_csv(args.train)
    if y is None:
        raise ValueError("Training CSV must contain TARGET.")
    x_train_raw, x_valid_raw, y_train, y_valid = split_by_time(raw, y, args.validation_period)

    preprocessor = FeaturePreprocessor(drop_time=not args.keep_time)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_valid = preprocessor.transform(x_valid_raw)
    preprocessor.save(artifacts / "preprocessor.joblib")

    ae = NormalAutoencoder(AutoencoderConfig(latent_dim=args.latent_dim, epochs=args.ae_epochs, seed=args.seed))
    ae.fit(x_train[y_train.to_numpy() == 0])
    x_train_hybrid = augment(x_train, ae)
    x_valid_hybrid = augment(x_valid, ae)
    ae.save(artifacts / "autoencoder.pt")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )
    params = {
        "n_estimators": [300, 600, 900],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 5, 15],
        "reg_lambda": [1.0, 5.0, 15.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(
        base, params, n_iter=args.n_iter, scoring="average_precision",
        cv=cv, n_jobs=args.n_jobs, refit=True, random_state=args.seed, verbose=1,
    )
    search.fit(x_train_hybrid, y_train)
    model = search.best_estimator_
    joblib.dump(model, artifacts / "xgb_model.joblib")
    np.save(artifacts / "validation_y.npy", y_valid.to_numpy())
    np.save(artifacts / "validation_pred.npy", model.predict_proba(x_valid_hybrid)[:, 1])
    metadata = {
        "validation_period": args.validation_period or str(x_valid_raw["TIME_PERIOD"].iloc[0]),
        "train_rows": len(x_train_raw), "validation_rows": len(x_valid_raw),
        "positive_rate_train": float(y_train.mean()), "scale_pos_weight": scale_pos_weight,
        "raw_feature_count": int(x_train.shape[1]), "hybrid_feature_count": int(x_train_hybrid.shape[1]),
        "best_params": search.best_params_,
        "validation_pr_auc": float(average_precision_score(y_valid, np.load(artifacts / "validation_pred.npy"))),
        "validation_roc_auc": float(roc_auc_score(y_valid, np.load(artifacts / "validation_pred.npy"))),
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--validation-period", default=None)
    parser.add_argument("--keep-time", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=80)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
