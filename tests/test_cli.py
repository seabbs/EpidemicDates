# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import csv
import runpy
import sys

import pytest

from onset_imputation import main
from onset_imputation.cli import build_parser
from onset_imputation.methods import METHODS


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.reps == 200
    assert args.seed == 20261027
    assert args.out == "output"
    assert args.scenarios is None
    assert not args.quiet


def test_main_writes_results(tmp_path, capsys):
    out = tmp_path / "sim"
    main(["--reps", "1", "--scenarios", "B0", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "done B0" in printed
    assert "forward mean" in printed
    with (out / "results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert {r["scenario"] for r in rows} == {"B0"}
    assert {r["method"] for r in rows} == set(METHODS)
    assert (out / "analytic.txt").exists()


def test_main_quiet(tmp_path, capsys):
    main(["--reps", "1", "--scenarios", "B0", "--out", str(tmp_path), "-q"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--reps", "0"],
        ["--reps", "x"],
        ["--scenarios", "Z"],
        ["--unknown"],
    ],
)
def test_main_rejects_bad_arguments(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err


def test_main_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--scenarios" in capsys.readouterr().out


def test_module_entry_point(tmp_path, monkeypatch, capsys):
    argv = ["onset-imputation", "--reps", "1", "--scenarios", "B0"]
    argv += ["--out", str(tmp_path), "--quiet"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("onset_imputation", run_name="__main__")
    assert (tmp_path / "results.csv").exists()
