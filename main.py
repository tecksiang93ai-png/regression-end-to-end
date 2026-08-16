# Standard library imports
import logging
import random

# Third-party imports
import numpy as np

# Local application imports
from src.data_preparation import DataPreparation, load_config, load_data
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

CONFIG_PATH = "./src/config.yaml"


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy so a run is reproducible end to end."""
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    # 1. Config (validated) + reproducibility.
    config = load_config(CONFIG_PATH)
    set_global_seed(config["random_seed"])

    # 2. Load raw data (schema-checked).
    df = load_data(config)

    # 3. Clean + engineer features.
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    # 4. Split into train / validation / test.
    trainer = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned_df)

    # 5. Train + tune every configured model, evaluate on validation.
    results = trainer.train_and_tune(X_train, y_train, X_val, y_val)

    # 6. Select the best by the configured validation metric.
    best_name = trainer.select_best(results)

    # 7. Refit the winner on train+val, then evaluate ONCE on the held-out test set.
    final_model = trainer.refit_on_train_val(
        results[best_name]["pipeline"], X_train, X_val, y_train, y_val
    )
    final_metrics = trainer.evaluate(final_model, X_test, y_test, f"{best_name} (final test)")

    # 8. Persist artifacts (model, comparison table, predictions, run manifest).
    trainer.save_artifacts(best_name, final_model, results, final_metrics, X_test, y_test)

    logging.info("Pipeline complete. Best model: %s", best_name)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Pipeline aborted: %s", exc)
        raise
