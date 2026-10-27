"""Stochastic simulation study of onset-date imputation for missing onsets.

An independent cross-check of the maths in imputation.js.
"""

from .simulation import (
    DESC,
    METHODS,
    SCEN,
    analytic,
    gamma_pmf,
    main,
    run_scenario,
    run_study,
    write_results,
)

__all__ = [
    "DESC",
    "METHODS",
    "SCEN",
    "analytic",
    "gamma_pmf",
    "main",
    "run_scenario",
    "run_study",
    "write_results",
]
