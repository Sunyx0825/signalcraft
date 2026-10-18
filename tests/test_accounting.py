import numpy as np
import pandas as pd
import pytest

from signalcraft.backtest import BacktestConfig, MarketData, rebalance, select_holdings, simulate
from signalcraft.signals import smooth_ranks


def market_and_signals():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "A",
            "close": [1.0, 2.0, 4.0, 4.0],
            "volume": 100.0,
            "change": 0.0,
        }
    )
    signals = pd.DataFrame({"datetime": dates, "instrument": "A", "score": 1.0})
    return MarketData.from_frame(prices), smooth_ranks(signals)


def test_entry_and_exit_costs_conserve_capital():
    config = BacktestConfig(holdings=1, buffer_rank=1, buy_bps=10, sell_bps=20)
    values, cash, cost, buys, sells = rebalance(
        np.array([0.0]), 1.0, [0], np.array([False]), 1, config
    )
    assert values[0] == pytest.approx(1 / 1.001)
    assert values.sum() + cash + cost == pytest.approx(1)
    target, cash, cost, buys, sells = rebalance(values, cash, [], np.array([False]), 0, config)
    assert target.sum() == 0
    assert cash == pytest.approx((1 / 1.001) * 0.998)


def test_signal_does_not_earn_same_day_or_execution_day_return():
    market, signals = market_and_signals()
    daily, _ = simulate(signals, market, "baseline", BacktestConfig(1, 1, 0, 0))
    assert daily["return"].tolist() == pytest.approx([0, 1, 0])
    assert daily.nav.iloc[-1] == 2
    assert daily.positions.iloc[-1] == 0


def test_frozen_position_keeps_value_and_a_slot():
    config = BacktestConfig(holdings=2, buffer_rank=3)
    values = np.array([0.5, 0, 0])
    selected, frozen, slots = select_holdings(
        np.array([3, 1, 2]), values, np.array([False, True, True]), config, False
    )
    assert selected == [1]
    assert slots == 1
    target, *_ = rebalance(values, 0.5, selected, frozen, slots, config)
    assert target[0] == values[0]


def test_buffer_retains_held_stock_within_boundary():
    config = BacktestConfig(holdings=1, buffer_rank=2)
    selected, _, _ = select_holdings(
        np.array([2, 1]), np.array([1, 0]), np.array([True, True]), config, True
    )
    assert selected == [0]
    selected, _, _ = select_holdings(
        np.array([3, 1]), np.array([1, 0]), np.array([True, True]), config, True
    )
    assert selected == [1]


def test_smoothing_has_no_future_dependence_and_keeps_missing_labels():
    dates = pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2)
    frame = pd.DataFrame(
        {"datetime": dates, "instrument": ["A", "B"] * 2, "score": [2, 1, 1, 2], "label": np.nan}
    )
    output = smooth_ranks(frame)
    assert len(output) == 4
    assert output.smooth.tolist() == pytest.approx([1, 0.5, 0.75, 0.75])
    frame.loc[frame.datetime == dates[-1], "score"] *= -10
    pd.testing.assert_frame_equal(output.iloc[:2], smooth_ranks(frame).iloc[:2])


def test_terminal_missing_price_is_not_silently_dropped():
    market, signals = market_and_signals()
    market.close.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="no further price"):
        simulate(signals, market, "baseline", BacktestConfig(1, 1, 0, 0))
