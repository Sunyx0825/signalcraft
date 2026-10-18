"""Turnover-aware signal processing and self-financing portfolio simulation."""

from .backtest import BacktestConfig, MarketData, simulate
from .signals import smooth_ranks

__all__ = ["BacktestConfig", "MarketData", "simulate", "smooth_ranks"]
