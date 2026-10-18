"""Reproduce all four strategies under three transaction-cost scenarios."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, MarketData, simulate
from .metrics import summarize
from .signals import smooth_ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    predictions_path = args.data / "predictions_test.csv.gz"
    prices_path = args.data / "prices.csv.gz"
    predictions = pd.read_csv(predictions_path, parse_dates=["datetime"])
    signals = smooth_ranks(predictions)
    market = MarketData.from_frame(pd.read_csv(prices_path, parse_dates=["datetime"]))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("baseline", "smooth", "buffer", "smooth_buffer"):
        for buy_bps, sell_bps in ((0, 0), (5, 15), (10, 20)):
            config = BacktestConfig(buy_bps=buy_bps, sell_bps=sell_bps)
            daily, holdings = simulate(signals, market, variant, config)
            row = {"variant": variant, "buy_bps": buy_bps, "sell_bps": sell_bps, **summarize(daily)}
            rows.append(row)
            tag = f"{variant}_{buy_bps}_{sell_bps}"
            daily.to_csv(args.output / f"daily_{tag}.csv", index=False, float_format="%.17g")
            if (buy_bps, sell_bps) == (5, 15):
                holdings.to_csv(args.output / f"holdings_{variant}.csv.gz", index=False)
            print(
                f"{tag}: net={row['annual_net_return']:.4%}, turnover={row['mean_daily_turnover']:.4%}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False, float_format="%.17g")
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (predictions_path, prices_path)
    }
    (args.output / "input_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
