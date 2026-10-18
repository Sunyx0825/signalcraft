"""Check a fresh experiment against the published numerical reference."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
expected = pd.read_csv(ROOT / "results/comparison.csv")
actual = pd.read_csv(ROOT / "output/comparison.csv")
pd.testing.assert_frame_equal(actual, expected, atol=1e-9, rtol=1e-9)
for path in sorted((ROOT / "output").glob("daily_*.csv")):
    daily = pd.read_csv(path)
    np.testing.assert_allclose((1 + daily["return"]).cumprod(), daily.nav, atol=1e-10)
    np.testing.assert_allclose(
        (1 + daily.gross_return_before_cost) * (1 - daily.cost_nav) - 1,
        daily["return"],
        atol=1e-12,
    )
    if daily.positions.iloc[-1] != 0:
        raise AssertionError(f"Unliquidated positions: {path.name}")
print("12 experiments reproduce the reference; daily accounting reconciles.")
