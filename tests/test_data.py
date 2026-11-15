import numpy as np
import pandas as pd
import pytest

from src.config import BANKS, SYNTHETIC_LABEL
from src.data import load_returns_csv, load_synthetic_dataset
from src.preprocessing import calculate_log_returns, clean_price_data
from src.synthetic_data import generate_synthetic_data, save_synthetic_data


def test_synthetic_generation_is_reproducible_and_positive():
    first = generate_synthetic_data(seed=41)
    second = generate_synthetic_data(seed=41)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert len(first[0]) >= 1200
    assert list(first[0].columns) == list(BANKS)
    assert (first[0] > 0).all().all()
    assert first[1].std().max() > first[1].std().min()
    assert first[3]["stress_period"].any()
    assert first[3]["data_status"].eq(SYNTHETIC_LABEL).all()
    assert first[0].attrs["data_status"] == SYNTHETIC_LABEL
    stress_volatility = first[1].loc[first[3]["stress_period"]].std().mean()
    normal_volatility = first[1].loc[~first[3]["stress_period"]].std().mean()
    assert stress_volatility > normal_volatility
    assert "MASI_SIM" in first[2]
    correlations = first[1].corr().to_numpy()
    off_diagonal = correlations[np.triu_indices_from(correlations, k=1)]
    assert off_diagonal.mean() > 0.1


def test_data_loader_writes_only_missing_canonical_files(tmp_path):
    save_synthetic_data(tmp_path)
    loaded = load_synthetic_dataset(tmp_path)
    assert len(loaded) == 4
    assert list(loaded[1].columns) == list(BANKS)
    assert len(list(tmp_path.glob("*.csv"))) == 4
    saved_prices = (tmp_path / "bank_prices.csv").read_bytes()
    save_synthetic_data(tmp_path)
    assert (tmp_path / "bank_prices.csv").read_bytes() == saved_prices


def test_data_loader_generates_an_initial_dataset_when_all_files_are_absent(tmp_path):
    loaded = load_synthetic_dataset(tmp_path)
    assert len(loaded) == 4
    assert len(list(tmp_path.glob("*.csv"))) == 4


def test_data_loader_refuses_partial_dataset_without_overwriting_existing_files(
    tmp_path,
):
    prices_path = tmp_path / "bank_prices.csv"
    prices_path.write_text("preserve this file", encoding="utf-8")
    original = prices_path.read_bytes()

    with pytest.raises(FileNotFoundError, match="refusing automatic regeneration"):
        load_synthetic_dataset(tmp_path)

    assert prices_path.read_bytes() == original
    assert list(tmp_path.glob("*.csv")) == [prices_path]


def test_save_refuses_partial_dataset_unless_force_is_explicit(tmp_path):
    prices_path = tmp_path / "bank_prices.csv"
    prices_path.write_text("preserve this file", encoding="utf-8")
    original = prices_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to regenerate"):
        save_synthetic_data(tmp_path)

    assert prices_path.read_bytes() == original
    assert list(tmp_path.glob("*.csv")) == [prices_path]


def test_force_explicitly_regenerates_partial_dataset(tmp_path):
    prices_path = tmp_path / "bank_prices.csv"
    prices_path.write_text("replace only with explicit force", encoding="utf-8")

    saved = save_synthetic_data(tmp_path, force=True)

    assert len(saved) == 4
    assert all(path.exists() for path in saved)
    assert "ATW_SIM" in pd.read_csv(prices_path).columns


def test_loader_rejects_csv_missing_synthetic_provenance(tmp_path):
    save_synthetic_data(tmp_path)
    path = tmp_path / "bank_prices.csv"
    frame = pd.read_csv(path)
    frame["data_status"] = "unverified"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="synthetic-data label"):
        load_synthetic_dataset(tmp_path)


def test_price_cleaning_and_log_returns_handle_duplicates_and_missing():
    dates = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])
    prices = pd.DataFrame({"bank": [110.0, 100.0, 121.0]}, index=dates)
    cleaned = clean_price_data(prices)
    assert len(cleaned) == 2
    returns = calculate_log_returns(cleaned)
    assert np.isclose(returns.iloc[0, 0], np.log(1.21))


def test_nonpositive_prices_are_rejected():
    prices = pd.DataFrame({"bank": [10.0, 0.0]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="strictly positive"):
        clean_price_data(prices)


def test_returns_do_not_bridge_a_missing_price_observation():
    dates = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"bank": [100.0, np.nan, 121.0, 133.1]}, index=dates)
    returns = calculate_log_returns(prices, align="outer", drop_missing=False)
    assert pd.isna(returns.loc[dates[2], "bank"])
    assert np.isclose(returns.loc[dates[3], "bank"], np.log(1.1))


def test_csv_import_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text("date,bank\n2024-01-01,0.01\n2024-01-01,-0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate dates"):
        load_returns_csv(path)


def test_csv_import_requires_an_explicit_synthetic_data_label(tmp_path):
    unlabeled = tmp_path / "unlabeled.csv"
    unlabeled.write_text("date,bank\n2024-01-01,0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic data"):
        load_returns_csv(unlabeled)

    labeled = tmp_path / "labeled.csv"
    labeled.write_text(
        f"date,bank,data_status\n2024-01-01,0.01,{SYNTHETIC_LABEL}\n",
        encoding="utf-8",
    )
    assert load_returns_csv(labeled).iloc[0, 0] == 0.01
