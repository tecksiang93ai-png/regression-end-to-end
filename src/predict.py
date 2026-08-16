"""
predict.py — inference interface for the HDB resale-price model.

Loads the trained pipeline saved by main.py and generates predictions for a new
CSV of transactions (same raw schema as the training data).

Usage
-----
    python -m src.predict --input data/new_transactions.csv --output predictions.csv

The model, config, and cleaning logic are reused exactly as in training, so the
input is cleaned and engineered identically before prediction.
"""

# Standard library imports
import argparse
import logging
from pathlib import Path

# Third-party imports
import joblib
import pandas as pd

# Local application imports
from src.data_preparation import DataPreparation, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_inference(input_csv: str, output_csv: str, config_path: str = "./src/config.yaml") -> pd.DataFrame:
    """Clean an input CSV and predict resale prices with the saved model."""
    config = load_config(config_path)

    model_path = Path(config["artifacts_dir"]) / "best_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path.resolve()}. Run main.py first."
        )
    model = joblib.load(model_path)

    if not Path(input_csv).is_file():
        raise FileNotFoundError(f"Input CSV not found: {Path(input_csv).resolve()}")
    raw = pd.read_csv(input_csv)

    prep = DataPreparation(config)
    cleaned = prep.clean_data(raw.copy())
    # Drop the target if the caller happened to include it — it is not an input.
    cleaned = cleaned.drop(columns=[config["target_column"]], errors="ignore")

    predictions = model.predict(cleaned)
    result = raw.copy()
    result["predicted_" + config["target_column"]] = predictions

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    logging.info("Wrote %d predictions to %s", len(result), Path(output_csv).resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict HDB resale prices for new data.")
    parser.add_argument("--input", required=True, help="Path to input CSV (raw schema).")
    parser.add_argument("--output", default="predictions.csv", help="Where to write predictions.")
    parser.add_argument("--config", default="./src/config.yaml", help="Path to config.yaml.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.input, args.output, args.config)
