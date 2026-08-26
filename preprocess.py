"""Feature engineering and leakage-safe tabular preprocessing.

The fitted FeaturePreprocessor is serialized and reused unchanged for
validation and production prediction rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


TENURE_COLUMNS = ("AVERAGE_ACCT_AGE1", "CREDIT_HISTORY_LENGTH1")
ROLLING_GROUPS = (
    ("ONEMNTH", "ONEMNTH_AVG"),
    ("TWOMNTH", "TWOMNTH_AVG"),
    ("THREEMNTH", "THREEMNTH_AVG"),
)


def parse_tenure_months(value: object) -> float:
    """Convert values such as '7yrs 6mon' to total months."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(-?\d+)\s*yrs?\s+(-?\d+)\s*mon", str(value).strip().lower())
    if match is None:
        return np.nan
    return float(int(match.group(1)) * 12 + int(match.group(2)))


def engineer_features(frame: pd.DataFrame, drop_time: bool = True) -> pd.DataFrame:
    """Apply deterministic transformations without fitting on the data."""
    df = frame.copy()

    for column in TENURE_COLUMNS:
        if column in df:
            df[f"{column}_MONTHS"] = df[column].map(parse_tenure_months)
            df = df.drop(columns=column)

    for prefix, output in ROLLING_GROUPS:
        columns = [
            f"{prefix}OUTSTANGBAL",
            f"{prefix}AVGMTD",
            f"{prefix}AVGQTD",
            f"{prefix}AVGYTD",
        ]
        available = [column for column in columns if column in df]
        if available:
            df[output] = df[available].mean(axis=1)
            df = df.drop(columns=available)

    criif = [f"CRIFF_{i}" for i in range(11, 67, 11)]
    available_criif = [column for column in criif if column in df]
    if available_criif:
        df["CRIFF_AVG"] = df[available_criif].mean(axis=1)
        df = df.drop(columns=available_criif)

    if drop_time:
        df = df.drop(columns=["TIME_PERIOD"], errors="ignore")

    return df.replace([np.inf, -np.inf], np.nan)


@dataclass
class FeaturePreprocessor:
    """Sklearn transformer wrapper with stable output feature names."""

    drop_time: bool = True
    missing_column_threshold: float = 0.995
    transformer: Optional[ColumnTransformer] = None
    feature_names_: Optional[np.ndarray] = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        df = engineer_features(frame, drop_time=self.drop_time)
        keep = df.columns[df.isna().mean() < self.missing_column_threshold]
        df = df.loc[:, keep]

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [column for column in df.columns if column not in numeric]
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
            ]
        )
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # sparse_output was renamed to sparse in older sklearn releases.
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                ("onehot", encoder),
            ]
        )

        self.transformer = ColumnTransformer(
            [
                ("numeric", numeric_pipe, numeric),
                ("categorical", categorical_pipe, categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.transformer.fit(df)
        self.feature_names_ = self.transformer.get_feature_names_out()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform().")
        df = engineer_features(frame, drop_time=self.drop_time)
        expected = []
        for name in self.transformer.feature_names_in_:
            expected.append(name)
        for name in expected:
            if name not in df:
                df[name] = np.nan
        return np.asarray(self.transformer.transform(df[expected]), dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePreprocessor":
        return joblib.load(path)


def load_csv(path: str | Path, target: str = "TARGET") -> tuple[pd.DataFrame, Optional[pd.Series]]:
    df = pd.read_csv(path, low_memory=False)
    y = df.pop(target) if target in df else None
    return df, y


def split_by_time(
    df: pd.DataFrame,
    y: pd.Series,
    validation_period: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split; defaults to the latest period in labeled data."""
    if "TIME_PERIOD" not in df:
        raise ValueError("TIME_PERIOD is required for the chronological split.")
    periods = df["TIME_PERIOD"].astype(str)
    month_order = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                   "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    ordered = sorted(
        periods.dropna().unique(),
        key=lambda value: (int(str(value)[-2:]), month_order.get(str(value)[:3], 0)),
    )
    holdout = validation_period or ordered[-1]
    train_mask = periods != holdout
    if not train_mask.any() or train_mask.all():
        raise ValueError(f"Invalid validation period {holdout!r}; available={ordered}.")
    return df.loc[train_mask], df.loc[~train_mask], y.loc[train_mask], y.loc[~train_mask]
