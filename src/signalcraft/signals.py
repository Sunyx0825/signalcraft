"""Causal smoothing of daily cross-sectional percentile ranks."""

import numpy as np
import pandas as pd


def smooth_ranks(predictions: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Smooth ranks across consecutive membership days; reset re-entering stocks."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    required = {"datetime", "instrument", "score"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(predictions.columns))}")
    if predictions.empty or predictions.duplicated(["datetime", "instrument"]).any():
        raise ValueError("Predictions must be nonempty and unique per stock and day.")
    if not np.isfinite(predictions.score).all():
        raise ValueError("Every stock needs a finite prediction, even when its label is missing.")
    previous = {}
    frames = []
    for _, group in predictions.groupby("datetime", sort=True):
        group = group.sort_values("instrument").copy()
        group["rank"] = group.score.rank(pct=True)
        group["smooth"] = [
            alpha * rank + (1 - alpha) * previous.get(stock, rank)
            for stock, rank in zip(group.instrument, group["rank"])
        ]
        previous = dict(zip(group.instrument, group.smooth))
        frames.append(group)
    return pd.concat(frames, ignore_index=True)
