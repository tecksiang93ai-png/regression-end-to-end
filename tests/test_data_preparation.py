"""Tests for cleaning, feature engineering, the preprocessor, and config/data loaders."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data_preparation import DataPreparation, load_config, load_data


def _branch(preprocessor, name):
    """Return the transformer for a named branch of a ColumnTransformer."""
    for n, transformer, _cols in preprocessor.transformers:
        if n == name:
            return transformer
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Feature-engineering helpers
# --------------------------------------------------------------------------- #
def test_convert_storey_range_midpoint():
    assert DataPreparation._convert_storey_range("07 TO 09") == 8.0
    assert DataPreparation._convert_storey_range("01 TO 03") == 2.0


def test_extract_lease_info_verbose_format():
    assert DataPreparation._extract_lease_info("70 years 03 months") == 70 * 12 + 3


def test_extract_lease_info_bare_number():
    assert DataPreparation._extract_lease_info("81") == 81 * 12


def test_extract_lease_info_missing_returns_nan():
    assert np.isnan(DataPreparation._extract_lease_info(np.nan))


def test_extract_lease_info_unparseable_returns_nan():
    assert np.isnan(DataPreparation._extract_lease_info("not a lease"))


def test_convert_storey_range_malformed_returns_nan():
    # Malformed band must not crash the cleaning run; it yields NaN for the imputer.
    assert np.isnan(DataPreparation._convert_storey_range("GROUND FLOOR"))


def test_fill_missing_names_recovers_from_id(config):
    df = pd.DataFrame({"town_id": [3, 3, 1], "town_name": [None, "TAMPINES", "ANG MO KIO"]})
    out = DataPreparation._fill_missing_names(df, "town_id", "town_name")
    assert out.loc[0, "town_name"] == "TAMPINES"     # recovered from the id map


# --------------------------------------------------------------------------- #
# clean_data end-to-end
# --------------------------------------------------------------------------- #
def test_clean_merges_four_room_typo(config, raw_df):
    """Regression test for the pandas copy-on-write bug: 'FOUR ROOM' MUST merge."""
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert "FOUR ROOM" not in clean["flat_type"].unique()
    assert "4 ROOM" in clean["flat_type"].unique()


def test_clean_fixes_negative_lease_commence(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert (clean["lease_commence_date"] < 0).sum() == 0


def test_clean_drops_exact_duplicates(config, raw_df):
    # raw_df has 40 unique rows + 1 appended exact duplicate.
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert len(clean) == 40


def test_clean_engineers_year_and_month(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert {"year", "month"}.issubset(clean.columns)
    assert clean["year"].between(2015, 2020).all()
    assert clean["month"].between(1, 12).all()


def test_clean_engineers_remaining_lease_months(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert "remaining_lease_months" in clean.columns
    assert clean["remaining_lease_months"].notna().all()


def test_clean_drops_identifier_and_raw_columns(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    for dropped in ["id", "town_id", "flatm_id", "remaining_lease", "block", "street_name"]:
        assert dropped not in clean.columns


def test_clean_recovers_missing_town_name(config, raw_df):
    clean = DataPreparation(config).clean_data(raw_df.copy())
    assert clean["town_name"].notna().all()


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def test_numeric_branch_imputes_and_scales(config):
    pre = DataPreparation(config).preprocessor
    num = _branch(pre, "num")
    assert isinstance(num.named_steps["imputer"], SimpleImputer)
    assert num.named_steps["imputer"].strategy == config["numeric_imputer_strategy"]
    assert isinstance(num.named_steps["scaler"], StandardScaler)


def test_nominal_branch_imputes_and_onehots_dense(config):
    pre = DataPreparation(config).preprocessor
    nom = _branch(pre, "nom")
    onehot = nom.named_steps["onehot"]
    assert isinstance(onehot, OneHotEncoder)
    assert onehot.handle_unknown == "ignore"
    assert onehot.sparse_output is False   # dense: required by HistGradientBoosting


def test_ordinal_branch_uses_configured_categories(config):
    pre = DataPreparation(config).preprocessor
    ordn = _branch(pre, "ord")
    enc = ordn.named_steps["ordinal"]
    assert isinstance(enc, OrdinalEncoder)
    assert enc.categories[0] == config["flat_type_categories"]


def test_preprocessor_imputes_all_nans(config, raw_df):
    """Missing values in numeric and categorical inputs must not survive transform."""
    prep = DataPreparation(config)
    clean = prep.clean_data(raw_df.copy())
    clean.loc[clean.index[0], "floor_area_sqm"] = np.nan
    clean.loc[clean.index[1], "town_name"] = np.nan
    X = clean.drop(columns=[config["target_column"]])
    out = prep.preprocessor.fit_transform(X)
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------- #
# Config + data loaders
# --------------------------------------------------------------------------- #
def test_load_config_returns_validated_dict(config):
    assert isinstance(config["models"], dict) and config["models"]
    assert config["selection_mode"] in ("max", "min")


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("./src/does_not_exist.yaml")


def test_load_config_missing_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("target_column: resale_price\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_load_data_missing_column_raises(config, tmp_path):
    incomplete = tmp_path / "data.csv"
    pd.DataFrame({"resale_price": [1, 2]}).to_csv(incomplete, index=False)
    cfg = dict(config, file_path=str(incomplete))
    with pytest.raises(ValueError):
        load_data(cfg)
