# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check the JavaScript module and the Python package agree.

Runs epidemic-dates.js in Node on the same inputs as the Python code.
Skipped if Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import epidemic_dates as ed

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "epidemic-dates.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is not installed"
)


def run_js(code, payload):
    """Run `code` in Node with `M` (the module) and `input` defined."""
    script = (
        f"const M = require({json.dumps(str(JS))});"
        "const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"process.stdout.write(JSON.stringify(({code})));"
    )
    out = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def linelist():
    """A simulated line list with a turn in growth, cut at day 140."""
    rng = np.random.default_rng(3)
    sp = ed.scenario_setup("C")
    onset, report, missing = ed.simulate("C", sp, rng)
    keep = report <= 140
    onset, report, missing = onset[keep], report[keep], missing[keep]
    return {
        "onset": [
            None if m else int(o) for o, m in zip(onset, missing, strict=True)
        ],
        "report": [int(r) for r in report],
        "cutoff": 140,
    }


@pytest.mark.parametrize("mean,cv", [(8, 1.0), (6.6, 0.92), (14, 0.7)])
def test_delay_pmf_matches(mean, cv):
    js = run_js(
        "Array.from(M.delayPmf(input.mean, input.cv))", {"mean": mean, "cv": cv}
    )
    js = np.array(js[: ed.DMAX + 1])
    js /= js.sum()
    py = ed.gamma_pmf(mean, mean * cv)
    assert np.allclose(js, py, atol=1e-8)


def test_linelist_backsampling_and_mean_match(linelist):
    js = run_js(
        "M.imputeLinelist({...input, maxDelay: 60, iterations: 0}).imputed",
        linelist,
    )
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = ed.impute_linelist(obs, linelist["report"], cutoff=140)["imputed"]
    assert np.allclose(js["backsampling"], py["backsampling"], atol=1e-9)
    assert np.allclose(js["mean"], py["mean"], atol=1e-9)


def test_linelist_em_agrees(linelist):
    js = run_js(
        "M.imputeLinelist({...input, maxDelay: 60}).imputed.gen", linelist
    )
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = ed.impute_linelist(obs, linelist["report"], cutoff=140)["imputed"]
    js_w = np.add.reduceat(np.array(js), np.arange(0, 140, 7))
    py_w = np.add.reduceat(py["em"], np.arange(0, 140, 7))
    busy = py_w >= 50
    assert np.allclose(js_w[busy], py_w[busy], rtol=0.02)


@pytest.mark.parametrize("given", [True, False])
def test_nowcast_point_matches(linelist, given):
    delay = ed.F_DEFAULT.tolist() if given else None
    js = run_js(
        "M.nowcastLinelist({...input, nSamples: 1, bootstrap: false})"
        ".point.map(x => (Number.isFinite(x) ? x : null))",
        {**linelist, "delay": delay},
    )
    js = np.array([np.nan if x is None else x for x in js], dtype=float)
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = ed.nowcast_linelist(
        obs,
        linelist["report"],
        cutoff=linelist["cutoff"],
        delay=delay,
        n_samples=1,
        bootstrap=False,
    )["point"]
    assert np.array_equal(np.isnan(js), np.isnan(py))
    ok = np.isfinite(py)
    assert np.allclose(js[ok], py[ok], rtol=1e-9, atol=1e-9)


def test_nowcast_quantiles_agree_roughly(linelist):
    js = run_js(
        "M.nowcastLinelist({...input, nSamples: 400}).weeklyQuantiles",
        linelist,
    )
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = ed.nowcast_linelist(
        obs, linelist["report"], cutoff=linelist["cutoff"], n_samples=400
    )["weekly_quantiles"]
    assert [r["target_end_date"] for r in js] == [
        r["target_end_date"] for r in py
    ]
    for a, b in zip(js, py, strict=True):
        if b["quantile"] == 0.5 and b["value"] > 50:
            assert abs(a["value"] / b["value"] - 1) < 0.05


FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "cross-language.json").read_text()
)


def _weekly_close(a, b, rtol=0.02, min_cases=50):
    """Weekly totals within `rtol` on weeks with at least `min_cases`."""
    a_w = np.add.reduceat(np.asarray(a), np.arange(0, len(a), 7))
    b_w = np.add.reduceat(np.asarray(b), np.arange(0, len(b), 7))
    busy = b_w >= min_cases
    assert busy.any()
    return np.allclose(a_w[busy], b_w[busy], rtol=rtol)


def _observed(case):
    return np.array(
        [np.nan if o is None else o for o in case["onset"]], dtype=float
    )


def test_python_matches_the_fixture_with_periods():
    case = FIXTURE["linelist_periods"]
    res = ed.impute_linelist(
        _observed(case),
        case["report"],
        cutoff=case["cutoff"],
        delay_period=case["delay_period"],
        delay_shrinkage=case["delay_shrinkage"],
    )
    for key in ("backsampling", "mean", "em"):
        assert np.allclose(res["imputed"][key], case["imputed"][key])
    assert res["delay_period_start"].tolist() == case["delay_period_start"]
    assert np.allclose(res["em_delay_by_period"], case["em_delay_by_period"])


def test_python_matches_the_fixture_with_a_delay_matrix():
    case = FIXTURE["linelist_delay_matrix"]
    res = ed.impute_linelist(
        _observed(case),
        case["report"],
        cutoff=case["cutoff"],
        delay=case["delay"],
    )
    for key in ("backsampling", "mean", "forward", "em"):
        assert np.allclose(res["imputed"][key], case["imputed"][key])


def test_js_matches_the_fixture_with_periods():
    case = FIXTURE["linelist_periods"]
    js = run_js(
        "M.imputeLinelist({...input, delayPeriod: input.delay_period,"
        " delayShrinkage: input.delay_shrinkage})",
        case,
    )
    ref = case["imputed"]
    assert np.allclose(js["imputed"]["backsampling"], ref["backsampling"])
    assert np.allclose(js["imputed"]["mean"], ref["mean"], atol=1e-9)
    assert js["delayPeriodStart"] == case["delay_period_start"]
    assert _weekly_close(js["imputed"]["gen"], ref["em"])


def test_js_matches_the_fixture_with_a_delay_matrix():
    case = FIXTURE["linelist_delay_matrix"]
    js = run_js("M.imputeLinelist(input).imputed", case)
    ref = case["imputed"]
    for js_key, key in (
        ("backsampling", "backsampling"),
        ("mean", "mean"),
        ("fwd", "forward"),
    ):
        assert np.allclose(js[js_key], ref[key], atol=1e-9)
    assert _weekly_close(js["gen"], ref["em"])


@pytest.mark.parametrize("variant", ["periods", "matrix"])
@pytest.mark.parametrize("method", ["em", "forward"])
def test_nowcast_point_matches_with_delays_that_change(variant, method):
    case = FIXTURE[
        "linelist_periods" if variant == "periods" else "linelist_delay_matrix"
    ]
    extra = (
        {"delay_period": case["delay_period"]}
        if variant == "periods"
        else {"delay": case["delay"]}
    )
    js_extra = (
        {"delayPeriod": case["delay_period"]}
        if variant == "periods"
        else {"delay": case["delay"]}
    )
    js_method = {"em": "gen", "forward": "fwd"}[method]
    js = run_js(
        "M.nowcastLinelist({...input, nSamples: 1, bootstrap: false})"
        ".point.map(x => (Number.isFinite(x) ? x : null))",
        {
            "onset": case["onset"],
            "report": case["report"],
            "cutoff": case["cutoff"],
            "method": js_method,
            **js_extra,
        },
    )
    js = np.array([np.nan if x is None else x for x in js], dtype=float)
    py = ed.nowcast_linelist(
        _observed(case),
        case["report"],
        cutoff=case["cutoff"],
        method=method,
        n_samples=1,
        bootstrap=False,
        **extra,
    )["point"]
    assert np.array_equal(np.isnan(js), np.isnan(py))
    ok = np.isfinite(py)
    assert np.allclose(js[ok], py[ok], rtol=1e-9, atol=1e-9)
