# SBI Hybrid Default/Fraud Detection

End-to-end tabular default/fraud prediction using a hybrid architecture:

```text
Feature engineering + preprocessing
                |
                v
PyTorch autoencoder trained on TARGET == 0
                |
                +--> latent embeddings
                +--> reconstruction error
                |
                v
Processed raw features + latent features + reconstruction error
                |
                v
XGBoost binary classifier
                |
                v
Risk score, top-K prioritization, and SHAP explanations
```

The project is designed for highly imbalanced financial data where the useful
business question is usually "which records should be investigated first?"
rather than overall accuracy.

## Dataset

The project expects these files in the repository root:

```text
HACKATHON_TRAINING_DATA.CSV
HACKATHON_PREDICTION_DATA.CSV
```

Observed dataset profile:

| Dataset | Rows | Columns | Target |
|---|---:|---:|---|
| Training | 327,741 | 139 | `TARGET` |
| Prediction | 191,693 | 138 | Not available |

Important columns:

| Column | Role |
|---|---|
| `TARGET` | Binary default/fraud label in the training data |
| `UNIQUE_ID` | Record identifier; preserved only for prediction output |
| `TIME_PERIOD` | Temporal partition used for validation |
| `AVERAGE_ACCT_AGE1` | Text tenure such as `7yrs 6mon` |
| `CREDIT_HISTORY_LENGTH1` | Text tenure such as `7yrs 6mon` |
| `INCOME_BAND1`, `AGREG_GROUP`, `PRODUCT_TYPE` | Categorical features |
| `LATEST_CR_DAYS`, `LATEST_DR_DAYS` | Numeric recency features |

The training data contains the periods `NOV24`, `DEC24`, and `JAN25`. The
prediction data contains `FEB25` and `MAR25`.

The observed training target distribution is:

```text
TARGET == 0: 292,301
TARGET == 1:  35,440
Positive rate: approximately 10.81%
```

The validation period `JAN25` has 110,002 rows and a positive rate of
approximately 10.06%.

## Modeling Process

### 1. Temporal data split

The default split is:

```text
Development: NOV24 + DEC24
Validation:  JAN25
Production prediction: FEB25 + MAR25
```

This is preferred over a purely random split because it better approximates
future deployment. `TIME_PERIOD` is used for partitioning and is removed from
the model features by default.

### 2. Feature engineering

Implemented in `preprocess.py`:

- Converts tenure strings such as `7yrs 6mon` into total months.
- Replaces positive and negative infinity with missing values.
- Creates row-wise monthly aggregates for the first three monthly account
  groups.
- Creates a combined `CRIFF_AVG` feature.
- Preserves numeric, categorical, binary flag, and missingness information.

The transformations are deterministic and are applied identically to
development, validation, and prediction data.

### 3. Preprocessing

The fitted preprocessing pipeline uses:

- Median imputation for numeric columns.
- Most-frequent imputation for categorical columns.
- Missingness indicators.
- `RobustScaler` for numeric features, reducing the impact of extreme
  financial-value outliers.
- One-hot encoding for categorical features.
- `handle_unknown="ignore"` for new production categories.

The preprocessor is fitted only on the development period and saved as:

```text
preprocessor.joblib
```

### 4. PyTorch autoencoder

The autoencoder is implemented in `autoencoder.py`.

```text
Input features
      |
      v
Dense(128) + ReLU
      |
      v
Dense(64) + ReLU
      |
      v
Latent bottleneck, default size 16
      |
      v
Dense(64) + ReLU
      |
      v
Dense(128) + ReLU
      |
      v
Reconstructed input
```

Training rules:

- Only rows with `TARGET == 0` are used.
- The objective is mean squared reconstruction error.
- AdamW optimization is used.
- Early stopping is applied using a normal-class validation subset.
- The autoencoder produces:
  - A latent vector `z`.
  - A sample-level reconstruction error.

The checkpoint is saved as:

```text
autoencoder.pt
```

### 5. Hybrid feature augmentation

For every row, the final XGBoost input is:

```text
hybrid_features =
    processed_raw_features
    + latent_embedding
    + reconstruction_error
```

The reconstruction error can identify records whose behavior differs from the
normal-class pattern learned by the autoencoder.

### 6. XGBoost classifier

The supervised classifier is an `XGBClassifier` configured with:

- Binary logistic objective.
- `aucpr` evaluation metric.
- Histogram tree method.
- `scale_pos_weight = negatives / positives`.
- Stratified 3-fold cross-validation.
- Randomized hyperparameter search.
- Average Precision, equivalent to PR-AUC, as the tuning objective.

The default search covers:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`

## Current Results

The current trained model used:

```json
{
  "validation_period": "JAN25",
  "train_rows": 217739,
  "validation_rows": 110002,
  "positive_rate_train": 0.11193676833272863,
  "scale_pos_weight": 7.933615065851557,
  "raw_feature_count": 257,
  "hybrid_feature_count": 274,
  "validation_pr_auc": 0.43677780839015296,
  "validation_roc_auc": 0.8436152437391535
}
```

Best XGBoost parameters from that run:

```text
subsample:         1.00
reg_lambda:        5.00
n_estimators:      900
min_child_weight: 5
max_depth:         5
learning_rate:     0.02
colsample_bytree:  0.85
```

Interpretation:

- Validation ROC-AUC of `0.8436` indicates good ranking discrimination.
- Validation PR-AUC of `0.4368` is strong for this imbalanced dataset.
- The `JAN25` positive-rate baseline is approximately `0.1006`, so the model's
  PR-AUC is roughly 4.34 times the random prevalence baseline.
- The model should be assessed operationally using Recall@Top-K, such as the
  percentage of all defaults captured by investigating the top 1%, 5%, or 10%
  of scored records.

The unlabeled `FEB25` and `MAR25` prediction data cannot be used to calculate
performance metrics until corresponding labels are available.

## Project Files

| File | Purpose |
|---|---|
| `preprocess.py` | Feature engineering, imputation, scaling, encoding, and temporal split |
| `autoencoder.py` | PyTorch MLP autoencoder and checkpoint handling |
| `train_pipeline.py` | Full preprocessing, autoencoder training, XGBoost tuning, and artifact saving |
| `evaluate.py` | PR-AUC, ROC-AUC, Recall@Top-K, SHAP, and prediction CSV generation |
| `run_experiments.py` | Automated sweep over autoencoder and XGBoost search parameters |
| `EDA.ipynb` | Original exploratory data analysis |
| `ml_model.ipynb` | Original classical ML experiments |
| `deep_learning_model.ipynb` | Original neural-network experiments |
| `requirements.txt` | Python dependencies |

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Training

Run a quick smoke experiment:

```powershell
python train_pipeline.py `
  --train HACKATHON_TRAINING_DATA.CSV `
  --artifacts artifacts_test `
  --ae-epochs 10 `
  --n-iter 2 `
  --n-jobs 2
```

Run the full default training process:

```powershell
python train_pipeline.py `
  --train HACKATHON_TRAINING_DATA.CSV `
  --artifacts artifacts `
  --n-iter 12 `
  --n-jobs -1
```

Use a specific validation period:

```powershell
python train_pipeline.py `
  --train HACKATHON_TRAINING_DATA.CSV `
  --validation-period JAN25 `
  --artifacts artifacts
```

Training creates:

```text
artifacts/
├── preprocessor.joblib
├── autoencoder.pt
├── xgb_model.joblib
├── validation_y.npy
├── validation_pred.npy
└── metadata.json
```

## Evaluation

Evaluate the labeled temporal holdout:

```powershell
python evaluate.py `
  --data HACKATHON_TRAINING_DATA.CSV `
  --artifacts artifacts `
  --shap
```

The output includes:

- Average Precision / PR-AUC.
- ROC-AUC.
- Recall@Top-1%.
- Recall@Top-5%.
- Recall@Top-10%.
- Optional top-20 mean absolute SHAP importances.

Generate predictions for the unlabeled prediction dataset:

```powershell
python evaluate.py `
  --data HACKATHON_PREDICTION_DATA.CSV `
  --artifacts artifacts `
  --output predictions.csv
```

The generated file has this format:

```text
UNIQUE_ID,PREDICTION
```

## Parameter Experiments

Run a small grid over autoencoder epochs, XGBoost search iterations, and
latent dimension:

```powershell
python run_experiments.py `
  --train HACKATHON_TRAINING_DATA.CSV `
  --artifacts-root experiments `
  --ae-epochs 40 80 `
  --n-iter 12 25 `
  --latent-dim 16 32 `
  --n-jobs 4
```

This performs eight runs and ranks them using validation PR-AUC. Results are
saved to:

```text
experiments/summary.json
```

`n_jobs` controls CPU parallelism and memory usage; it does not improve model
quality and should normally be kept fixed during experiments.

## Recommended Improvements

The next improvements should be measured against the current temporal
validation result:

1. Compare against a raw-feature-only XGBoost baseline.
2. Add financial ratios such as utilization, installment-to-limit, and
   overdue-account ratios.
3. Add monthly trend, volatility, and recency features across all 12 monthly
   groups.
4. Tune `reg_alpha`, `gamma`, and `max_delta_step`.
5. Add probability calibration if scores will be interpreted as probabilities.
6. Select an operating threshold based on investigation capacity and the cost
   of missed defaults.
7. Use cross-fitted autoencoder features inside each validation fold for a
   stricter unbiased estimate.
8. Monitor performance separately by `PRODUCT_TYPE`, `TIME_PERIOD`, and
   missingness pattern.

Do not select a model only because it has the highest training score. A model
with very high training PR-AUC and unchanged validation PR-AUC is overfitting.
The primary selection metric should remain temporal validation PR-AUC, with
Recall@Top-K as the operational metric.
