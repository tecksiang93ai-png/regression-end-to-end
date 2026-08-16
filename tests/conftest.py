"""Shared pytest fixtures for the HDB resale-price test suite."""
import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable so `from src...` works when pytest is run
# from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preparation import load_config  # noqa: E402

CONFIG_PATH = ROOT / "src" / "config.yaml"


@pytest.fixture(scope="session")
def config():
    """The real, validated project configuration."""
    return load_config(str(CONFIG_PATH))


@pytest.fixture
def fast_config(config):
    """A copy of the config restricted to fast models, for training smoke tests."""
    cfg = copy.deepcopy(config)
    cfg["models"] = {
        "dummy": {"estimator": "DummyRegressor", "grid": {}},
        "linear_regression": {"estimator": "LinearRegression", "grid": {}},
        "ridge": {"estimator": "Ridge", "grid": {"regressor__alpha": [1.0, 10.0]}},
    }
    cfg["cv"] = 2
    return cfg


@pytest.fixture
def raw_df():
    """A small synthetic frame mirroring the raw CSV schema, with deliberate
    data-quality issues (a duplicate row, a negative lease year, the 'FOUR ROOM'
    typo, both remaining_lease formats, and a missing town/flat-model name)."""
    rows = []
    towns = [(1, "ANG MO KIO"), (2, "BEDOK"), (3, "TAMPINES")]
    fms = [(1, "Improved"), (2, "Model A"), (3, "New Generation")]
    flat_types = ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
    storeys = ["01 TO 03", "07 TO 09", "10 TO 12", "13 TO 15"]
    for i in range(40):
        tid, tname = towns[i % 3]
        fid, fname = fms[i % 3]
        lease = "81" if i % 2 == 0 else "70 years 03 months"
        rows.append({
            "id": i + 1,
            "month": f"2018-{(i % 12) + 1:02d}",
            "flat_type": flat_types[i % 4],
            "block": f"{100 + i}",
            "street_name": f"STREET {i % 5}",
            "storey_range": storeys[i % 4],
            "floor_area_sqm": 70.0 + (i % 40),
            "lease_commence_date": 1990 + (i % 20),
            "remaining_lease": lease,
            "resale_price": 300000.0 + i * 5000,
            "town_id": tid,
            "flatm_id": fid,
            "town_name": tname,
            "flatm_name": fname,
        })
    df = pd.DataFrame(rows)
    # Inject issues:
    df.loc[0, "flat_type"] = "FOUR ROOM"          # label typo -> must merge to 4 ROOM
    df.loc[1, "lease_commence_date"] = -2005       # corrupt sign -> abs()
    df.loc[2, "town_name"] = None                  # missing name -> back-fill from town_id
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # exact duplicate row
    return df
