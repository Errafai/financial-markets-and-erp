import json

import pandas as pd

from src.config import SYNTHETIC_LABEL
from src.export import export_indicators


def test_export_writes_csv_and_iso_json_with_synthetic_label(tmp_path):
    indicators = pd.DataFrame(
        {
            "bank_code": ["ATW_SIM"],
            "covar": [-0.03],
            "delta_covar": [-0.01],
            "anomaly": [False],
        },
        index=pd.DatetimeIndex(["2025-01-02"], name="date"),
    )
    csv_path, json_path = export_indicators(indicators, tmp_path)
    record = json.loads(json_path.read_text(encoding="utf-8"))[0]
    assert record["date"] == "2025-01-02"
    assert record["bank_code"] == "ATW_SIM"
    assert record["data_status"] == SYNTHETIC_LABEL
    assert csv_path.exists()
