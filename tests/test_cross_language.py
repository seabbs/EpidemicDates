# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Check the JavaScript module and the Python package agree.

Runs onset-imputation.js in Node on the same inputs as the Python code.
Skipped if Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import onset_imputation as oi

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "onset-imputation.js"

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
    sp = oi.scenario_setup("C")
    onset, report, missing = oi.simulate("C", sp, rng)
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
    js = np.array(js[: oi.DMAX + 1])
    js /= js.sum()
    py = oi.gamma_pmf(mean, mean * cv)
    assert np.allclose(js, py, atol=1e-8)


def test_linelist_hub_and_mean_match(linelist):
    js = run_js(
        "M.imputeLinelist({...input, maxDelay: 60, iterations: 0}).imputed",
        linelist,
    )
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = oi.impute_linelist(obs, linelist["report"], cutoff=140)["imputed"]
    assert np.allclose(js["hub"], py["hub"], atol=1e-9)
    assert np.allclose(js["mean"], py["mean"], atol=1e-9)


def test_linelist_em_agrees(linelist):
    js = run_js(
        "M.imputeLinelist({...input, maxDelay: 60}).imputed.gen", linelist
    )
    obs = np.array(
        [np.nan if o is None else o for o in linelist["onset"]], dtype=float
    )
    py = oi.impute_linelist(obs, linelist["report"], cutoff=140)["imputed"]
    js_w = np.add.reduceat(np.array(js), np.arange(0, 140, 7))
    py_w = np.add.reduceat(py["em"], np.arange(0, 140, 7))
    busy = py_w >= 50
    assert np.allclose(js_w[busy], py_w[busy], rtol=0.02)
