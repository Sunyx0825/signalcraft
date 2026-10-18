"""Portfolio metrics computed from the self-financing ledger."""

import numpy as np
import pandas as pd


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    returns = daily["return"]
    deviation = returns.std(ddof=1)
    peaks = daily.nav.cummax().clip(lower=1.0)
    return {
        "annual_net_return": float(daily.nav.iloc[-1] ** (252 / len(daily)) - 1),
        "mean_daily_turnover": float(daily.turnover.mean()),
        "sharpe": float(returns.mean() / deviation * np.sqrt(252)) if deviation > 0 else 0.0,
        "max_drawdown": float((daily.nav / peaks - 1).min()),
        "end_nav": float(daily.nav.iloc[-1]),
        "days": int(len(daily)),
        "ending_positions": int(daily.positions.iloc[-1]),
    }
