# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Optional JAX versions of the core pieces, for use in NumPyro models.

Needs the `jax` extra: `pip install "epidemic-dates[jax]"` (or
`uv sync --extra jax`).
Every function except `tabulate_linelist` is pure, jittable and
differentiable.
`tabulate_linelist` runs in NumPy, outside `jit`, and builds the arrays
that `linelist_loglik` needs.

Importing this module turns on 64-bit floats in JAX
(`jax.config.update("jax_enable_x64", True)`), so results match the
NumPy functions to about 1e-12.
This is a global JAX setting and affects other JAX code in the same
process.
`gamma_pmf` and `backsampling_allocate` are drop-in JAX versions of the
NumPy functions of the same names.
"""

from __future__ import annotations

try:
    import jax
except ImportError as err:  # pragma: no cover - depends on the install
    raise ImportError(
        "epidemic_dates.jax needs JAX. Install the `jax` extra, for "
        'example `pip install "epidemic-dates[jax]"` or '
        "`uv sync --extra jax`."
    ) from err

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call, unused-ignore]

from ._core import (  # noqa: E402
    backsampling_allocate,
    convolve_reports,
    delay_pmf,
    gamma_pmf,
    linelist_loglik,
    reported_fraction,
)
from ._model import numpyro_linelist_model  # noqa: E402
from ._tabulate import LinelistArrays, tabulate_linelist  # noqa: E402

__all__ = [
    "LinelistArrays",
    "backsampling_allocate",
    "convolve_reports",
    "delay_pmf",
    "gamma_pmf",
    "linelist_loglik",
    "numpyro_linelist_model",
    "reported_fraction",
    "tabulate_linelist",
]
