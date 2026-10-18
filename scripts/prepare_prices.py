"""Convert a local Qlib price snapshot into the compact research input panel."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_field(root, calendar, stock, field):
    path = root / "features" / stock.lower() / f"{field}.day.bin"
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2 or not np.isfinite(values[0]) or values[0] != int(values[0]):
        raise ValueError(f"Invalid Qlib field header: {path}")
    start = int(values[0])
    if start < 0 or start + len(values) - 1 > len(calendar):
        raise ValueError(f"Qlib field falls outside the calendar: {path}")
    aligned = np.full(len(calendar), np.nan)
    aligned[start : start + len(values) - 1] = values[1:]
    return pd.Series(aligned, index=calendar)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlib-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions_test.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, parse_dates=["datetime"])
    calendar = pd.DatetimeIndex(
        pd.to_datetime((args.qlib_dir / "calendars/day.txt").read_text().splitlines())
    )
    selected_dates = calendar[
        (calendar >= frame.datetime.min()) & (calendar <= frame.datetime.max())
    ]
    records = []
    for stock in sorted(frame.instrument.unique()):
        rows = pd.DataFrame(
            {
                name: read_field(args.qlib_dir, calendar, stock, name).reindex(selected_dates)
                for name in ("close", "volume", "change")
            }
        )
        rows["instrument"] = stock
        rows.index.name = "datetime"
        records.append(rows.reset_index())
    panel = pd.concat(records, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "prices.csv.gz"
    panel.to_csv(
        output, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0}
    )
    membership = pd.read_csv(
        args.qlib_dir / "instruments/csi300.txt",
        sep="\t",
        header=None,
        names=["instrument", "start", "end"],
        parse_dates=["start", "end"],
    )
    extra, missing = 0, 0
    for date, group in frame.groupby("datetime"):
        expected = set(
            membership.loc[(membership.start <= date) & (membership.end >= date), "instrument"]
        )
        actual = set(group.instrument)
        extra += len(actual - expected)
        missing += len(expected - actual)
    if extra or missing:
        raise ValueError(f"Historical membership mismatch: extra={extra}, missing={missing}")
    audit = {
        "rows": len(panel),
        "signal_rows": len(frame),
        "signal_days": frame.datetime.nunique(),
        "historical_membership_extra": extra,
        "historical_membership_missing": missing,
        "prices_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source": "https://github.com/chenditc/investment_data/releases/tag/2024-12-07",
    }
    (args.output / "price_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
