# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Command-line entry point, `epidemic-dates`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .scenarios import SCEN
from .study import analytic, run_study, write_results

__all__ = ["build_parser", "main"]


def _positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for `epidemic-dates`.

    Returns
    -------
    argparse.ArgumentParser
        Parser with options `--reps`, `--seed`, `--out`, `--scenarios`
        and `--quiet`.
    """
    parser = argparse.ArgumentParser(
        prog="epidemic-dates",
        description="Simulation study of onset-date imputation bias.",
    )
    parser.add_argument(
        "--reps",
        type=_positive_int,
        default=200,
        help="replicates per scenario (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261027,
        help="base random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="output",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=list(SCEN),
        default=None,
        metavar="NAME",
        help=f"scenarios to run, from {', '.join(SCEN)} (default: all)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not print progress or the analytic checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the simulation study from the command line.

    Runs `run_study`, writes results with `write_results` and prints the
    analytic checks.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments. Defaults to `sys.argv[1:]`. Options are `--reps`
        (default 200), `--seed` (default 20261027), `--out` (default
        `output`), `--scenarios` (default all) and `--quiet`.

    Raises
    ------
    SystemExit
        If the arguments are invalid, or for `--help`.
    """
    args = build_parser().parse_args(argv)
    rows = run_study(
        args.reps, args.seed, args.scenarios, verbose=not args.quiet
    )
    write_results(rows, args.out)
    if not args.quiet:
        print("\n".join(analytic()))
