"""Regenerate frozen predictions using the upstream MASTER model and checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .sampler import load

EXPECTED_HASHES = {
    "master.py": "e5e798952c9636c8c43fc91cff51a6dbbebc009b99d16a1ae090be32d0744fe4",
    "base_model.py": "116322431b83194234390a5929cd98883afb8fac2b1eaffb3d01556b9f5d691a",
    "csi300_opensource_0.pkl": "581443312928ac27fab1c94482cea9e3e0a250f50948fe17e88d8ab5fa188a59",
    "csi300_dl_test.pkl": "d05194f2b8d0a1d94e79a13abc2702e1c16c9d8e514ead3ea1b41d4de07cfd95",
}


def verify_hash(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != EXPECTED_HASHES[path.name]:
        raise ValueError(f"Unexpected {path.name}; use the documented upstream snapshot.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official MASTER checkout")
    parser.add_argument("--sampler", type=Path, required=True, help="Official csi300_dl_test.pkl")
    parser.add_argument("--output", type=Path, default=Path("output/predictions_test.csv.gz"))
    args = parser.parse_args()
    checkpoint = args.source / "model/csi300_opensource_0.pkl"
    for path in (
        args.source / "master.py",
        args.source / "base_model.py",
        checkpoint,
        args.sampler,
    ):
        verify_hash(path)
    # Import only after confirming the documented source and artifact hashes.
    import torch

    sys.path.insert(0, str(args.source.resolve()))
    from master import MASTER

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = MASTER(
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        beta=5,
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    sampler = load(args.sampler)
    index = sampler.get_index()
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("The model input index must be sorted and unique.")
    records = []
    with torch.inference_mode():
        for day, positions in pd.Series(np.arange(len(index)), index=index).groupby(
            level="datetime"
        ):
            batch = sampler.batch(positions.to_numpy())
            features = batch[:, :, :-1]
            if not np.isfinite(features).all():
                raise ValueError(f"Nonfinite model features on {day}.")
            scores = model(torch.from_numpy(features.copy()).float()).numpy()
            # Missing future labels are retained: they cannot determine eligibility.
            frame = pd.DataFrame(
                {"score": scores, "label": batch[:, -1, -1]}, index=positions.index
            )
            frame["rank"] = frame.score.rank(pct=True)
            records.append(frame)
    output = pd.concat(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, compression={"method": "gzip", "mtime": 0})
    print(
        json.dumps(
            {
                "rows": len(output),
                "missing_labels": int(output.label.isna().sum()),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
