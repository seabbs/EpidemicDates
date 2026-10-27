import importlib
import sys

import pytest


def test_jax_layer_needs_the_jax_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "epidemic_dates.jax" or name.startswith(
            "epidemic_dates.jax."
        ):
            monkeypatch.delitem(sys.modules, name)
    # A None entry makes `import jax` raise ImportError.
    monkeypatch.setitem(sys.modules, "jax", None)
    with pytest.raises(ImportError, match="`jax` extra"):
        importlib.import_module("epidemic_dates.jax")
