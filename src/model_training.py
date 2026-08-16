# Standard library imports
import json
import logging
from pathlib import Path
from typing import Any, Dict

# Related third-party imports
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Registry mapping the estimator names used in config.yaml to their classes.
ESTIMATORS = {
    "DummyRegressor": DummyRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}
# Estimators that accept a random_state, so we can seed them for reproducibility.
_SEEDED = {
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
}


class ModelTraining:
    """Train, tune, select, and evaluate regression models on HDB resale data.

    The set of models, their hyper-parameter grids, the selection policy, and the
    output location all come from ``config``; nothing is hard-coded here.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor
        self.seed = int(config.get("random_seed", 42))

    def split_data(self, df: pd.DataFrame):
        """Split into 80% train / 10% validation / 10% test using the config seed."""
        logging.info("Splitting data into train/validation/test.")
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.seed
        )
        logging.info(
            "Split sizes -> train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _make_pipeline(self, estimator_name: str) -> Pipeline:
        """Build a preprocess->regressor pipeline for a named estimator."""
        if estimator_name not in ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator_name}'. "
                f"Known: {sorted(ESTIMATORS)}"
            )
        kwargs: Dict[str, Any] = {}
        if estimator_name in _SEEDED:
            kwargs["random_state"] = self.seed
        model = ESTIMATORS[estimator_name](**kwargs)
        return Pipeline(steps=[("preprocessor", self.preprocessor), ("regressor", model)])

    def train_and_tune(self, X_train, y_train, X_val, y_val) -> Dict[str, Dict[str, Any]]:
        """Fit every model in the config. Models with a non-empty grid are tuned
        with cross-validated grid search; the rest are fit with defaults. Each is
        evaluated on the validation set. Returns a per-model result dict."""
        cv = self.config.get("cv", 5)
        scoring = self.config.get("scoring", "r2")
        results: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.config["models"].items():
            grid = spec.get("grid") or {}
            pipeline = self._make_pipeline(spec["estimator"])
            if grid:
                logging.info("Tuning %s over %d-fold CV ...", name, cv)
                search = GridSearchCV(
                    pipeline, grid, cv=cv, scoring=scoring, n_jobs=-1
                )
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                best_params = search.best_params_
                cv_best = float(search.best_score_)
            else:
                logging.info("Fitting %s (no tuning) ...", name)
                pipeline.fit(X_train, y_train)
                fitted = pipeline
                best_params = {}
                cv_best = None
            results[name] = {
                "pipeline": fitted,
                "val_metrics": self.evaluate(fitted, X_val, y_val, f"{name} (val)"),
                "best_params": best_params,
                "cv_best_score": cv_best,
            }
        return results

    def select_best(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Pick the winning model by the configured validation metric/mode."""
        metric = self.config["selection_metric"]
        mode = self.config["selection_mode"]
        chooser = max if mode == "max" else min
        best = chooser(results, key=lambda n: results[n]["val_metrics"][metric])
        logging.info("Best model on validation (%s %s): %s", mode, metric, best)
        return best

    def refit_on_train_val(self, fitted_pipeline, X_train, X_val, y_train, y_val):
        """Refit the winning model (with its tuned params) on train+val before the
        single final test evaluation, so it learns from all non-test data."""
        X = pd.concat([X_train, X_val])
        y = pd.concat([y_train, y_val])
        model = clone(fitted_pipeline)
        model.fit(X, y)
        logging.info("Refit best model on train+val (%d rows).", len(X))
        return model

    def evaluate(self, model, X, y, label: str) -> Dict[str, float]:
        """Compute MAE/MSE/RMSE/R2 and log them under a label."""
        pred = model.predict(X)
        metrics = {
            "MAE": float(mean_absolute_error(y, pred)),
            "MSE": float(mean_squared_error(y, pred)),
            "RMSE": float(root_mean_squared_error(y, pred)),
            "R2": float(r2_score(y, pred)),
        }
        logging.info(
            "%s -> MAE=%.0f RMSE=%.0f R2=%.4f", label, metrics["MAE"], metrics["RMSE"], metrics["R2"]
        )
        return metrics

    def save_artifacts(
        self, best_name, final_model, results, final_metrics, X_test, y_test
    ) -> None:
        """Persist the fitted pipeline, a model-comparison table, test predictions,
        and a run manifest (best model, params, metrics, seed, config snapshot)."""
        out = Path(self.config["artifacts_dir"])
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(final_model, out / "best_model.joblib")

        rows = []
        for name, r in results.items():
            row = {"model": name}
            row.update({f"val_{k}": v for k, v in r["val_metrics"].items()})
            row["cv_best_score"] = r["cv_best_score"]
            rows.append(row)
        pd.DataFrame(rows).sort_values("val_R2", ascending=False).to_csv(
            out / "model_comparison.csv", index=False
        )

        preds = pd.DataFrame(
            {"y_true": y_test.to_numpy(), "y_pred": final_model.predict(X_test)}
        )
        preds.to_csv(out / "test_predictions.csv", index=False)

        manifest = {
            "best_model": best_name,
            "best_params": results[best_name]["best_params"],
            "selection_metric": self.config["selection_metric"],
            "selection_mode": self.config["selection_mode"],
            "final_test_metrics": final_metrics,
            "random_seed": self.seed,
            "config_snapshot": self.config,
        }
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        logging.info("Artifacts written to %s", out.resolve())
