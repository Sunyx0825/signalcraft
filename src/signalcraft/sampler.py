"""Minimal, inert TSDataSampler state loader and exact sampling semantics.

Only named NumPy/Pandas reconstruction objects and the Qlib sampler state are
allowed. No Qlib module or application code is executed during unpickling.
Sliding-window semantics follow the official Qlib TSDataSampler. This archive
uses its older date-major layout with start_idx/end_idx, reconstructed directly
from the serialized date, instrument, and row-to-array mappings.
"""

import importlib
import pickle

import numpy as np


class Sampler:
    def get_index(self):
        return self.data_index[self.start_idx : self.end_idx].reorder_levels(
            ["datetime", "instrument"]
        )

    def __len__(self):
        return int(self.end_idx - self.start_idx)

    def batch(self, positions):
        indices = []
        for p in positions:
            row, col = self.idx_map[p + self.start_idx]
            idx = self.idx_arr[max(row - self.step_len + 1, 0) : row + 1, col].copy()
            if len(idx) < self.step_len:
                idx = np.r_[np.full(self.step_len - len(idx), np.nan), idx]
            if self.fillna_type in ("ffill", "ffill+bfill"):
                for i in range(1, len(idx)):
                    if np.isnan(idx[i]):
                        idx[i] = idx[i - 1]
                if self.fillna_type == "ffill+bfill":
                    for i in range(len(idx) - 2, -1, -1):
                        if np.isnan(idx[i]):
                            idx[i] = idx[i + 1]
            elif self.fillna_type != "none":
                raise ValueError(f"Unsupported fill method: {self.fillna_type}")
            indices.append(np.nan_to_num(idx, nan=self.nan_idx).astype(int))
        return self.data_arr[np.stack(indices)]


ALLOW = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.series", "Series"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.datetimes", "DatetimeIndex"),
    ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
    ("pandas._libs.tslibs.timestamps", "Timestamp"),
    ("pandas._libs.tslibs.nattype", "__nat_unpickle"),
    ("builtins", "slice"),
    ("builtins", "set"),
    ("collections", "OrderedDict"),
    ("datetime", "date"),
    ("datetime", "datetime"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("qlib.data.dataset", "TSDataSampler"):
            return Sampler
        if (module, name) not in ALLOW:
            raise pickle.UnpicklingError(f"Non-allowlisted global: {module}.{name}")
        return getattr(importlib.import_module(module), name)


def load(path):
    with open(path, "rb") as f:
        obj = RestrictedUnpickler(f).load()
    if not isinstance(obj, Sampler):
        raise ValueError("Expected a Qlib time-series sampler.")
    if obj.data_arr.ndim != 2 or obj.data_arr.shape[1] != 222 or obj.step_len != 8:
        raise ValueError("Unexpected feature layout or sequence length.")
    return obj
