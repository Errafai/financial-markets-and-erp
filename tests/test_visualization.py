import pandas as pd

from src.config import REAL_LABEL
from src.visualization import plot_series


def test_plot_labels_synthetic_data_and_saves_image(tmp_path):
    values = pd.Series(
        [0.01, -0.02, 0.005],
        index=pd.date_range("2025-01-01", periods=3),
        name="return",
    )
    output = tmp_path / "synthetic_returns.png"
    figure = plot_series(values, "Simulated returns", "Log return", output)
    assert REAL_LABEL in figure.axes[0].get_title()
    assert output.exists()
