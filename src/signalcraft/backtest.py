"""Daily close-marked accounting with next-day execution and explicit costs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    holdings: int = 30
    buffer_rank: int = 45
    buy_bps: float = 5
    sell_bps: float = 15
    daily_move_limit: float = 0.095

    def __post_init__(self):
        if self.holdings < 1 or self.buffer_rank < self.holdings:
            raise ValueError("Require 1 <= holdings <= buffer_rank.")
        if not 0 <= self.buy_bps < 10000 or not 0 <= self.sell_bps < 10000:
            raise ValueError("Costs must be finite and in [0, 10000) basis points.")
        if not 0 < self.daily_move_limit < 1:
            raise ValueError("daily_move_limit must be in (0, 1).")


@dataclass
class MarketData:
    close: pd.DataFrame
    volume: pd.DataFrame
    change: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MarketData":
        required = {"datetime", "instrument", "close", "volume", "change"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing price fields: {required - set(frame.columns)}")
        if frame.duplicated(["datetime", "instrument"]).any():
            raise ValueError("Duplicate stock/date prices.")
        matrices = [
            frame.pivot(index="datetime", columns="instrument", values=field).sort_index()
            for field in ("close", "volume", "change")
        ]
        if (matrices[0] <= 0).any().any():
            raise ValueError("Observed close prices must be positive; use NaN for missing prices.")
        return cls(*matrices)


def select_holdings(
    ranks: np.ndarray,
    values: np.ndarray,
    tradable: np.ndarray,
    config: BacktestConfig,
    use_buffer: bool,
) -> tuple[list[int], np.ndarray, int]:
    """Frozen positions keep their capital and slots; fill the remaining slots."""
    frozen = (values > 1e-14) & ~tradable
    slots = max(config.holdings - int(frozen.sum()), 0)
    order = np.argsort(np.nan_to_num(ranks, nan=np.inf), kind="stable")
    available = tradable & np.isfinite(ranks)
    selected = []
    if use_buffer:
        selected = [
            j
            for j in order
            if available[j] and values[j] > 1e-14 and ranks[j] <= config.buffer_rank
        ][:slots]
    chosen = set(selected)
    for j in order:
        if len(selected) >= slots:
            break
        if available[j] and j not in chosen:
            selected.append(int(j))
            chosen.add(j)
    return selected, frozen, slots


def rebalance(
    values: np.ndarray,
    cash: float,
    selected: list[int],
    frozen: np.ndarray,
    slots: int,
    config: BacktestConfig,
) -> tuple[np.ndarray, float, float, float, float]:
    """Solve equal target values after paying fees, preserving frozen holdings.

    Bisection brackets post-cost capital. The equation is monotone because both
    transaction cost rates are below one; expensive trades cannot create cash.
    """
    before = float(values.sum() + cash)

    def targets_and_cost(capital):
        target = np.where(frozen, values, 0.0)
        if selected and slots:
            target[selected] = max(capital - values[frozen].sum(), 0.0) / slots
        delta = target - values
        buys = float(np.maximum(delta, 0).sum())
        sells = float(np.maximum(-delta, 0).sum())
        cost = (buys * config.buy_bps + sells * config.sell_bps) / 10000
        return target, cost, buys, sells

    low, high = 0.0, before
    for _ in range(80):
        capital = (low + high) / 2
        target, cost, buys, sells = targets_and_cost(capital)
        if capital + cost > before:
            high = capital
        else:
            low = capital
    target, cost, buys, sells = targets_and_cost((low + high) / 2)
    ending_cash = before - cost - float(target.sum())
    if ending_cash < -1e-10:
        raise ArithmeticError("Rebalance spends more than available capital.")
    return target, max(ending_cash, 0.0), cost, buys, sells


def simulate(
    signals: pd.DataFrame,
    market: MarketData,
    variant: str,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the daily ledger and end-of-day holdings for one strategy.

    Signal t is executed at t+1 close. The last price date is a liquidation day.
    Fractional shares, no leverage, and zero cash interest are assumed.
    """
    if variant not in {"baseline", "smooth", "buffer", "smooth_buffer"}:
        raise ValueError(f"Unknown strategy: {variant}")
    if signals.empty or signals.duplicated(["datetime", "instrument"]).any():
        raise ValueError("Signals must be nonempty and unique per stock/day.")
    stocks = sorted(signals.instrument.unique())
    groups = {day: group.set_index("instrument") for day, group in signals.groupby("datetime")}
    calendar = market.close.index
    signal_dates = sorted(groups)
    if not set(signal_dates).issubset(calendar):
        raise ValueError("Every signal date must exist in the market calendar.")
    start = calendar.get_loc(signal_dates[0]) + 1
    stop = calendar.get_loc(signal_dates[-1]) + 1
    dates = calendar[start:stop]
    if len(dates) < 2:
        raise ValueError("Need at least two execution dates.")
    close = market.close.reindex(columns=stocks)
    marked = close.ffill()
    previous = marked.shift(1)
    last_valid = close.apply(pd.Series.last_valid_index)
    values = np.zeros(len(stocks))
    cash = previous_nav = 1.0
    ledger, holdings = [], []
    score_column = "smooth" if "smooth" in variant else "score"
    for position, day in enumerate(dates):
        held = values > 1e-14
        returns = (marked.loc[day] / previous.loc[day] - 1).to_numpy()
        if (held & (last_valid < day).to_numpy()).any():
            raise ValueError(f"Held stock has no further price data on {day}; resolve delisting.")
        if not np.isfinite(returns[held]).all():
            raise ValueError(f"Cannot mark a held stock on {day}.")
        values *= 1 + np.nan_to_num(returns, nan=0.0)
        before = float(values.sum() + cash)
        current_close = close.loc[day].to_numpy()
        volume = market.volume.reindex(columns=stocks).loc[day].to_numpy()
        change = market.change.reindex(columns=stocks).loc[day].to_numpy()
        tradable = (
            np.isfinite(current_close)
            & (current_close > 0)
            & np.isfinite(volume)
            & (volume > 0)
            & np.isfinite(change)
            & (np.abs(change) < config.daily_move_limit)
        )
        frozen = held & ~tradable
        if position == len(dates) - 1:
            signal_day, selected, slots = pd.NaT, [], 0
        else:
            signal_day = calendar[calendar.get_loc(day) - 1]
            if signal_day not in groups:
                raise ValueError(f"No signal for previous trading date {signal_day}.")
            ranks = (
                groups[signal_day][score_column]
                .rank(ascending=False, method="first")
                .reindex(stocks)
                .to_numpy()
            )
            selected, frozen, slots = select_holdings(
                ranks, values, tradable, config, "buffer" in variant
            )
        target, cash, cost, buys, sells = rebalance(values, cash, selected, frozen, slots, config)
        nav = float(target.sum() + cash)
        if not np.isclose(nav, before - cost, atol=1e-11, rtol=0):
            raise ArithmeticError("Portfolio and cash fail the accounting identity.")
        if not np.array_equal(target[frozen], values[frozen]):
            raise ArithmeticError("A frozen position was traded.")
        ledger.append(
            {
                "datetime": day,
                "signal_datetime": signal_day,
                "nav": nav,
                "return": nav / previous_nav - 1,
                "gross_return_before_cost": before / previous_nav - 1,
                "cost_nav": cost / before,
                "buy_nav": buys / before,
                "sell_nav": sells / before,
                "turnover": (buys + sells) / (2 * before),
                "cash_weight": cash / nav,
                "positions": int((target > 1e-14).sum()),
                "frozen_positions": int(frozen.sum()),
            }
        )
        for j in np.flatnonzero(target > 1e-14):
            holdings.append({"datetime": day, "instrument": stocks[j], "weight": target[j] / nav})
        values, previous_nav = target, nav
    return pd.DataFrame(ledger), pd.DataFrame(holdings)
