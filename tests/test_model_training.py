"""Tests for the model-training layer: factory, splitting, leakage, selection, metrics."""
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


def test_train_and_tune_returns_metrics_for_each_model(trainer_and_data, fast_config):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test) = trainer_and_data
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    assert set(results) == set(fast_config["models"])
    for r in results.values():
        assert set(r["val_metrics"]) == {"MAE", "MSE", "RMSE", "R2"}


def test_select_best_returns_a_configured_model(trainer_and_data, fast_config):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test) = trainer_and_data
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best = trainer.select_best(results)
    assert best in fast_config["models"]
    # A fitted model should beat the dummy baseline on R2.
    assert results[best]["val_metrics"]["R2"] >= results["dummy"]["val_metrics"]["R2"]


def test_refit_on_train_val_produces_predictions(trainer_and_data):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test) = trainer_and_data
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best = trainer.select_best(results)
    final = trainer.refit_on_train_val(results[best]["pipeline"], X_train, X_val, y_train, y_val)
    preds = final.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_metric_keys(trainer_and_data):
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test) = trainer_and_data
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)
    best = trainer.select_best(results)
    metrics = trainer.evaluate(results[best]["pipeline"], X_test, y_test, "test")
    assert set(metrics) == {"MAE", "MSE", "RMSE", "R2"}
