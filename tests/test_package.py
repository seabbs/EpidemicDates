# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import doctest
import importlib
from importlib.resources import files

import pytest

import onset_imputation as oi

MODULES = [
    "onset_imputation",
    "onset_imputation.cli",
    "onset_imputation.delays",
    "onset_imputation.linelist",
    "onset_imputation.methods",
    "onset_imputation.scenarios",
    "onset_imputation.study",
]


def test_version_is_a_string():
    assert isinstance(oi.__version__, str)
    assert oi.__version__


def test_all_names_exist_and_are_documented():
    for name in oi.__all__:
        obj = getattr(oi, name)
        if callable(obj) and name != "__version__":
            assert obj.__doc__, name


@pytest.mark.parametrize("module", MODULES)
def test_module_all_names_exist(module):
    mod = importlib.import_module(module)
    assert mod.__doc__
    for name in getattr(mod, "__all__", []):
        assert hasattr(mod, name), f"{module}.{name}"


@pytest.mark.parametrize("module", MODULES)
def test_doctests(module):
    result = doctest.testmod(
        importlib.import_module(module), optionflags=doctest.ELLIPSIS
    )
    assert result.failed == 0


def test_py_typed_marker_is_shipped():
    assert files("onset_imputation").joinpath("py.typed").is_file()
