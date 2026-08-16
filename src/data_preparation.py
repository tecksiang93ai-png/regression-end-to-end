# Standard library imports
import logging
import re
from pathlib import Path
from typing import Any, Dict

# Related third-party imports
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Keys that must be present in config.yaml for the pipeline to run.
REQUIRED_CONFIG_KEYS = [
    "file_path",
    "target_column",
    "random_seed",
    "numerical_features",
    "nominal_features",
    "ordinal_features",
    "passthrough_features",
    "flat_type_categories",
    "numeric_imputer_strategy",
    "categorical_imputer_strategy",
    "val_test_size",
    "val_size",
    "selection_metric",
    "selection_mode",
    "cv",
    "scoring",
    "models",
    "artifacts_dir",
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML config, failing early with a clear message.

    Raises
    ------
    FileNotFoundError : the config file does not exist.
    ValueError        : the YAML is malformed or a required key is missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse config YAML at {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Config is missing required key(s): {missing}")
    if config["selection_mode"] not in ("max", "min"):
        raise ValueError("selection_mode must be 'max' or 'min'.")
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("config['models'] must be a non-empty mapping of models.")
    logging.info("Configuration loaded and validated (%d models).", len(config["models"]))
    return config


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the raw CSV declared in config, verifying it exists and has the
    target plus the raw columns the cleaner depends on."""
    path = Path(config["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    df = pd.read_csv(path)
    required_raw = {
        config["target_column"], "month", "flat_type", "storey_range",
        "remaining_lease", "lease_commence_date", "town_id", "town_name",
        "flatm_id", "flatm_name",
    }
    missing = required_raw - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected column(s): {sorted(missing)}")
    # Sanity-check that the key numeric columns really are numeric (a common CSV
    # corruption is a stray text value turning a whole column to object dtype).
    for col in (config["target_column"], "floor_area_sqm", "lease_commence_date"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Expected numeric column '{col}' but got dtype {df[col].dtype}.")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logging.info("Loaded raw data: %d rows x %d columns (%.1f MB).", *df.shape, mem_mb)
    return df


class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting data cleaning.")
        df = df.drop_duplicates()
        # Direct reassignment (not chained `df[col].replace(..., inplace=True)`),
        # which silently no-ops under pandas' copy-on-write and left 'FOUR ROOM'
        # unmerged, sending 1,824 rows to the ordinal encoder's unknown bucket.
        df["flat_type"] = df["flat_type"].replace("FOUR ROOM", "4 ROOM")
        df["lease_commence_date"] = df["lease_commence_date"].abs()
        df["storey_range"] = df["storey_range"].apply(self._convert_storey_range)
        df = self._fill_missing_names(df, "town_id", "town_name")
        df = self._fill_missing_names(df, "flatm_id", "flatm_name")
        df.drop(columns=["id", "town_id", "flatm_id"], inplace=True)
        df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m")
        df["year"] = df["year_month"].dt.year
        df["month"] = df["year_month"].dt.month
        df.drop(columns=["year_month"], inplace=True)
        df["remaining_lease_months"] = df["remaining_lease"].apply(
            self._extract_lease_info
        )
        df.drop(columns=["remaining_lease", "block", "street_name"], inplace=True)
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        num_strategy = self.config.get("numeric_imputer_strategy", "median")
        cat_strategy = self.config.get("categorical_imputer_strategy", "most_frequent")
        # Impute first so no NaN reaches the scaler/encoders. remaining_lease_months
        # can be missing when a lease string fails to parse, and the recovered
        # town/flat-model names can still be missing if an id has no known mapping.
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=num_strategy)),
                ("scaler", StandardScaler()),
            ]
        )
        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=cat_strategy)),
                # Dense output: HistGradientBoosting requires dense X, and the
                # one-hot width here (~59 cols) is small enough that dense is cheap.
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[self.config["flat_type_categories"]],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor

    @staticmethod
    def _convert_storey_range(storey_range: str) -> float:
        """Convert a storey band like '07 TO 09' to its numeric midpoint (8.0).

        Malformed input (not an 'A TO B' pair) returns NaN rather than raising, so
        one bad row can't abort the whole cleaning run; the imputer fills it later.
        """
        try:
            low, high = str(storey_range).split(" TO ")
            return (int(low) + int(high)) / 2
        except (ValueError, AttributeError):
            return np.nan

    @staticmethod
    def _fill_missing_names(
        df: pd.DataFrame, id_column: str, name_column: str
    ) -> pd.DataFrame:
        """Back-fill missing values in ``name_column`` from an id->name mapping
        learned from the rows where the name is present (e.g. recover a missing
        `town_name` from its `town_id`)."""
        missing_names = df[name_column].isna()
        name_mapping = (
            df[[id_column, name_column]]
            .dropna()
            .drop_duplicates()
            .set_index(id_column)[name_column]
            .to_dict()
        )
        df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(
            name_mapping
        )
        return df

    @staticmethod
    def _extract_lease_info(lease_str) -> float:
        """Parse a remaining-lease string into whole months.

        Handles both stored formats: the verbose '70 years 03 months' and the bare
        '81' (years only). Missing or unparseable input returns ``np.nan`` — a
        consistent numeric sentinel that the pipeline's imputer then fills — rather
        than a silent 0, which would understate the lease.
        """
        if pd.isna(lease_str):
            return np.nan
        text = str(lease_str).strip()
        years_match = re.search(r"(\d+)\s*years?", text)
        months_match = re.search(r"(\d+)\s*months?", text)
        number_match = re.match(r"^\d+$", text)
        if years_match:
            years = int(years_match.group(1))
        elif number_match:
            years = int(number_match.group(0))
        else:
            return np.nan
        months = int(months_match.group(1)) if months_match else 0
        return float(years * 12 + months)