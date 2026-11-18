import numpy as np
import pandas as pd

from src.real_data import REAL_LABEL, load_real_bank_prices


def test_real_loader_aligns_all_five_bank_files(tmp_path):
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    for bank in ("ATW", "BCP", "BOA", "CIH", "CDM"):
        frame = pd.DataFrame(
            {
                "Session": dates.strftime("%d/%m/%Y"),
                "Last Price": np.linspace(100, 120, len(dates)),
            }
        )
        frame.to_csv(tmp_path / f"Cours_{bank}_test.csv", sep=";", index=False)

    prices = load_real_bank_prices(tmp_path)

    assert prices.shape == (120, 5)
    assert list(prices.columns) == ["ATW", "BCP", "BOA", "CIH", "CDM"]
    assert prices.index.is_monotonic_increasing
    assert np.isfinite(prices.to_numpy(dtype=float)).all()
