import io
import pickle

import pytest

from signalcraft.inference import verify_hash
from signalcraft.sampler import RestrictedUnpickler


def test_inference_rejects_changed_source_before_import(tmp_path):
    source = tmp_path / "master.py"
    source.write_text("raise RuntimeError('must never execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected master.py"):
        verify_hash(source)


def test_pickle_cannot_resolve_arbitrary_python_functions():
    # GLOBAL resolves a callable but does not call it; even that lookup is denied.
    payload = b"cbuiltins\neval\n."
    with pytest.raises(pickle.UnpicklingError, match="Non-allowlisted"):
        RestrictedUnpickler(io.BytesIO(payload)).load()
