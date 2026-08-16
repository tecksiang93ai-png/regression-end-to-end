"""Tests for the model-training layer: factory, splitting, leakage, selection, metrics."""
import copy

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


@pytest.fixture
def trainer_and_data(fast_config, raw_df):
    prep = DataPreparation(fast_config)
    clean = prep.clean_data(raw_df.copy())
    trainer = ModelTraining(fast_config, prep.preprocessor)
    split = trainer.split_data(clean)
    return trainer, split


@pytest.fixture
def trained(trainer_and_data):
    """Train once and share the results across tests that need a fitted set."""
    trainer, split = trainer_and_data
    X_train, X_val, X_test, y_train, y_val, y_test = split
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    return trainer, split, results


def test_make_pipeline_unknown_estimator_raises(fast_config):
    prep = DataPreparation(fast_config)
    trainer = ModelTraining(fast_config, prep.preprocessor)
    with pytest.raises(ValueError):
        trainer._make_pipeline("NotARealModel")


def test_make_pipeline_seeds_stochastic_model(fast_config):
    prep = DataPreparation(fast_config)
    trainer = ModelTraining(fast_config, prep.preprocessor)
    pipe = trainer._make_pipeline("RandomForestRegressor")
    assert isinstance(pipe, Pipeline)
    rf = pipe.named_steps["regressor"]
    assert isinstance(rf, RandomForestRegressor)
    assert rf.random_state == fast_config["random_seed"]


def test_split_has_no_target_leakage(fast_config, trainer_and_data):
    _, (X_train, X_val, X_test, y_train, y_val, y_test) = trainer_and_data
    target = fast_config["target_column"]
    for X in (X_train, X_val, X_test):
        assert target not in X.columns


def test_split_sizes_partition_the_data(trainer_and_data):
    _, (X_train, X_val, X_test, *_) = trainer_and_data
    assert len(X_train) + len(X_val) + len(X_test) == 40
    assert len(X_train) > len(X_val) and len(X_train) > len(X_test)


def test_train_and_tune_returns_metrics_for_each_model(trained, fast_config):
    trainer, split, results = trained
    assert set(results) == set(fast_config["models"])
    for r in results.values():
        assert set(r["val_metrics"]) == {"MAE", "MSE", "RMSE", "R2"}
        assert "cv_mean" in r and "cv_std" in r   # fold-level reporting captured


def test_select_best_returns_a_configured_model(trained, fast_config):
    trainer, split, results = trained
    best = trainer.select_best(results)
    assert best in fast_config["models"]
    # A fitted model should beat the dummy baseline on R2.
    assert results[best]["val_metrics"]["R2"] >= results["dummy"]["val_metrics"]["R2"]


def test_select_best_is_deterministic_under_ties(fast_config):
    """When the primary metric ties, the secondary metric + name decide — never
    dict order. Two models with equal R2 but different RMSE resolve predictably."""
    prep = DataPreparation(fast_config)
    trainer = ModelTraining(fast_config, prep.preprocessor)
    results = {
        "model_b": {"val_metrics": {"R2": 0.9, "RMSE": 200.0, "MAE": 1, "MSE": 1}},
        "model_a": {"val_metrics": {"R2": 0.9, "RMSE": 100.0, "MAE": 1, "MSE": 1}},
    }
    assert trainer.select_best(results) == "model_a"   # lower RMSE wins the tie


def test_refit_on_train_val_produces_predictions(trained):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test), results = trained
    best = trainer.select_best(results)
    final = trainer.refit_on_train_val(results[best]["pipeline"], X_train, X_val, y_train, y_val)
    preds = final.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_metric_keys(trained):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test), results = trained
    best = trainer.select_best(results)
    metrics = trainer.evaluate(results[best]["pipeline"], X_test, y_test, "test")
    assert set(metrics) == {"MAE", "MSE", "RMSE", "R2"}


def test_save_artifacts_writes_expected_files(fast_config, raw_df, tmp_path):
    cfg = copy.deepcopy(fast_config)
    cfg["artifacts_dir"] = str(tmp_path)
    prep = DataPreparation(cfg)
    clean = prep.clean_data(raw_df.copy())
    trainer = ModelTraining(cfg, prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(clean)
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best = trainer.select_best(results)
    final = trainer.refit_on_train_val(results[best]["pipeline"], X_train, X_val, y_train, y_val)
    metrics = trainer.evaluate(final, X_test, y_test, "test")
    trainer.save_artifacts(best, final, results, metrics, X_test, y_test)
    for fname in ("best_model.joblib", "model_comparison.csv", "test_predictions.csv",
                  "run_manifest.json", "environment.txt"):
        assert (tmp_path / fname).is_file(), f"missing artifact: {fname}"
