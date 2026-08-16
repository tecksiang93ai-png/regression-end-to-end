# HDB Resale Price Prediction for Cataria Property Solutions

## Project Description

Cataria Property Solutions aims to improve the accuracy and consistency of HDB resale
property valuations through machine learning. Traditional valuation methods rely heavily on
manual assessment and agent experience, which can be inconsistent and slow.

This project develops a regression pipeline that predicts HDB resale prices from historical
transaction data, providing data-driven valuations to support property consultants.

---

# Project Structure

```
.
├── data/
│   └── resale_transactions.csv     # raw transaction data
├── src/
│   ├── config.yaml                 # all pipeline settings (no hard-coded values)
│   ├── data_preparation.py         # config/data loaders, cleaning, feature eng, preprocessor
│   ├── model_training.py           # model factory, tuning, selection, evaluation, artifacts
│   └── predict.py                  # inference on new data with the saved model
├── tests/                          # pytest suite (cleaning, preprocessing, training)
├── reports/                        # persisted EDA summaries (cardinality, VIF, outliers)
├── eda.ipynb                       # exploratory data analysis
├── main.py                         # end-to-end pipeline entry point
├── requirements.txt                # pinned dependencies
└── README.md
```

---

# Prerequisites & Installation

* Python 3.11

```bash
git clone <repository_url>
cd regression-end-to-end

python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies are **version-pinned** in `requirements.txt` for reproducibility
(`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `joblib`).

---

# Usage

## Train

```bash
python main.py
python main.py --config ./src/config.yaml --log-level DEBUG
```

This validates the config, loads and cleans the data, trains and tunes every configured
model, selects the best on the validation set (with a deterministic tie-break), refits it on
train+validation, evaluates once on the held-out test set, and writes artifacts to
`./artifacts/`. Artifacts are saved under both stable names (`best_model.joblib`) and
run-versioned names (`best_model_<model>_<timestamp>.joblib`) so successive runs don't
overwrite each other; `environment.txt` captures the exact package versions used.

## Predict on new data

```bash
python main.py --mode predict --input data/new_transactions.csv --output predictions.csv
# equivalently:
python -m src.predict --input data/new_transactions.csv --output predictions.csv
```

Loads the saved model and scores a new CSV (same raw schema, validated before cleaning),
reusing the exact cleaning and feature-engineering logic from training.

## Run the tests

```bash
python -m pytest tests/ -q
```

## Configure

All behaviour is driven by `./src/config.yaml` — the random seed, feature lists, imputation
strategies, the split ratios, the models and their hyper-parameter grids, the model-selection
metric and tie-break, an optional feature-selection stage, and the artifacts location. The
config is validated on load, so a missing or invalid key fails fast with a clear message.

---

# Data Dictionary

Raw columns in `data/resale_transactions.csv`:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `id` | int | Row identifier (dropped; duplicates removed) |
| `month` | str `YYYY-MM` | Transaction month → engineered into `year` + `month` |
| `flat_type` | str | Flat size (`1 ROOM`…`EXECUTIVE`, `MULTI-GENERATION`); ordinal |
| `block`, `street_name` | str | Address labels (dropped: high-cardinality / leakage risk) |
| `storey_range` | str `AA TO BB` | Floor band → numeric midpoint |
| `floor_area_sqm` | float | Floor area in m² (strongest predictor) |
| `lease_commence_date` | int (year) | Lease start year (negative values corrected via abs) |
| `remaining_lease` | str | Two formats → engineered `remaining_lease_months` (int) |
| `town_id` / `town_name` | int / str | Town (id used to back-fill missing names, then dropped) |
| `flatm_id` / `flatm_name` | int / str | Flat model (same id back-fill treatment) |
| `resale_price` | float | **Target** — transacted resale price (SGD) |

Engineered features fed to the model: `floor_area_sqm`, `remaining_lease_months`,
`lease_commence_date`, `year` (numeric); `month`, `town_name`, `flatm_name` (nominal);
`flat_type` (ordinal); `storey_range` midpoint (passthrough).

---

# Pipeline Flow

1. **Config loading & validation** — `load_config` checks all required keys are present.
2. **Data loading** — `load_data` verifies the file exists and carries the expected columns.
3. **Cleaning** — drop duplicates; merge the `FOUR ROOM` label into `4 ROOM`; fix corrupt
   (negative) `lease_commence_date`; back-fill missing `town_name`/`flatm_name` from their ids.
4. **Feature engineering** — `storey_range` → band midpoint; `month` → numeric `year` + `month`;
   `remaining_lease` (both text formats) → `remaining_lease_months`. High-cardinality address
   fields (`block`, `street_name`) and id columns are dropped to avoid leakage/noise.
5. **Preprocessing** (inside a `ColumnTransformer`/`Pipeline`, fit on training data only):
   impute → scale numerics; impute → one-hot encode nominals; impute → ordinal-encode
   `flat_type`; pass through the numeric `storey_range`.
6. **Splitting** — 80% train / 10% validation / 10% test, seeded for reproducibility.
7. **Training & tuning** — every model in the config is fit; those with a grid are tuned with
   cross-validated grid search.
8. **Selection & final evaluation** — the best model on validation is refit on train+val, then
   evaluated once on the test set.
9. **Artifacts** — the fitted pipeline, a model-comparison table, test predictions, and a run
   manifest are saved.

---

# Key EDA Findings

The exploratory analysis (`eda.ipynb`) surfaced the data-quality issues the cleaning step
fixes (4,223 duplicate rows, 9,394 negative lease years, missing town/flat-model names, the
`FOUR ROOM` typo, and a mixed-format `remaining_lease` field) and the main drivers of price:

* **`floor_area_sqm`** is the strongest single predictor (r ≈ 0.64).
* **Storey height** and **remaining lease** are the next most useful numeric features.
* **`flat_type`** tracks price almost monotonically (1-ROOM → Multi-Generation), supporting an
  ordinal encoding, and **`town`** captures a large location effect.
* The target is right-skewed with genuine high-value outliers, and `lease_commence_date` is
  strongly collinear with `remaining_lease_months` — both reasons to favour tree ensembles.

---

# Feature Handling

| Feature                | Treatment                          | Reason                                             |
| ---------------------- | ---------------------------------- | -------------------------------------------------- |
| town_name, flatm_name  | Impute → One-Hot Encoding          | Unordered location/model categories                |
| flat_type              | Impute → Ordinal Encoding          | Preserve the natural flat-size ordering            |
| storey_range           | Convert to band midpoint (numeric) | Preserve floor-height ordering cheaply             |
| floor_area_sqm         | Impute → Standard Scaling          | Standardise the strongest numeric predictor        |
| remaining_lease        | Feature engineering → months (int) | Unify the two text formats into a number           |
| year, month            | Feature engineering from `month`   | Capture temporal / seasonal effects                |
| Missing town/flat names | Back-fill from id mapping          | Recover names without dropping rows                |
| Duplicate records      | Removed                            | Prevent bias                                       |
| block, street_name, id | Removed                            | High-cardinality identifiers → leakage/noise risk  |

---

# Models

A **`DummyRegressor`** (predict-the-mean) provides a naive baseline. **Linear / Ridge / Lasso**
give interpretable linear benchmarks. **RandomForest** and **HistGradientBoosting** capture the
non-linear feature interactions that dominate housing prices. Models with a hyper-parameter
grid are tuned with cross-validated grid search; the best on validation is selected.

---

# Model Evaluation

Metrics: MAE, MSE, RMSE, R² (validation shown for model comparison; the winner is re-evaluated
once on the held-out test set).

| Model                  | Validation R² |
| ---------------------- | ------------- |
| **RandomForest**       | **0.952**     |
| HistGradientBoosting   | 0.949         |
| Lasso                  | 0.858         |
| Linear Regression      | 0.858         |
| Ridge                  | 0.858         |
| Dummy (baseline)       | ~0.000        |

## Final Selected Model: RandomForest

| Metric | Test result |
| ------ | ----------- |
| R²     | 0.953       |
| RMSE   | ~$31,735    |
| MAE    | ~$22,376    |

Moving from linear models (R² ≈ 0.86) to a tuned RandomForest lifts test R² to **0.95** — the
non-linear models clearly capture interactions the linear models cannot, while the Dummy
baseline (R² ≈ 0) confirms the models add real predictive value.

---

# Limitations & Next Steps

**Limitations**

* The data covers **2015–2019 only**; the model should not be extrapolated to later periods
  without retraining, as HDB prices have since shifted.
* `lease_commence_date` and `remaining_lease_months` are strongly collinear (high VIF), so
  linear-model coefficients are not reliable for inference — the tree models are used instead.
* `block` / `street_name` micro-location is intentionally dropped; very localised price effects
  within a town are therefore not captured.
* The negative-lease correction assumes a sign-flip corruption; if some values are wrong in
  other ways, `abs()` will not catch them.

**Next steps**

* Add newer transactions and a time-based (rather than random) validation split.
* Try target-encoding or a grouped location feature to recover some micro-location signal
  without the leakage risk of raw address one-hot encoding.
* Enable the optional feature-selection stage and compare, and add SHAP-based feature
  importance to the reporting.

---

# Deployment Considerations

* **Integration** — the saved pipeline (`artifacts/best_model.joblib`) can be served via the
  `src/predict.py` inference interface behind an API or dashboard.
* **Monitoring** — property markets drift; track live prediction error and retrain when it
  degrades. The seeded, config-driven pipeline makes retraining reproducible.
* **Business impact** — a consistent, data-driven valuation to support faster, more reliable
  decisions for Cataria Property Solutions' consultants.
