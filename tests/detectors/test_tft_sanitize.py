"""TFT feature-column sanitization (pytorch-forecasting rejects '.' in names)."""
import numpy as np
import polars as pl

from gulfstream.detectors.tft import sanitize_feature_columns_for_tft


def test_sanitize_replaces_dots_in_feature_names():
    n = 5
    df = pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), eager=True),
            "CAC40:CAP.PA": np.random.rand(n),
            "DAX30:SAP.DE_vol": np.random.rand(n),
        }
    )
    out = sanitize_feature_columns_for_tft(df)
    feat_cols = [c for c in out.columns if c != "date"]
    assert all("." not in c for c in feat_cols)
    assert "CAC40:CAP_PA" in feat_cols
    assert "DAX30:SAP_DE_vol" in feat_cols
