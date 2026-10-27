// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// nowcastLinelist in epidemic-dates.js: a nowcast with uncertainty.
import { test } from "node:test";
import assert from "node:assert/strict";
import { M, LONG } from "./helpers.mjs";

const CUTOFF = 110;
const L = 60;

/** Poisson draw: Knuth below a mean of 30, rounded normal above. */
function rpois(lam, rng) {
  if (lam <= 0) return 0;
  if (lam > 30) {
    const u = rng() || 1e-12,
      v = rng();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * z));
  }
  const E = Math.exp(-lam);
  let k = 0,
    p = 1;
  do {
    k++;
    p *= rng();
  } while (p > E);
  return k - 1;
}

/**
 * A Poisson line list from a scenario with growth that turns, cut at
 * `cutoff`. Returns the eventual onsets of every case with a delay up to
 * `L`, and the true delay pmf.
 */
function simulate(seed, cutoff = CUTOFF) {
  const set = {
    ...LONG,
    knots: M.knotsFrom([
      [0, 0.05],
      [60, 0.05],
      [100, -0.02]
    ]),
    O0: 4
  };
  const sc = M.scenario({ ...M.DEFAULTS, ...set });
  const rng = M.makeRng(seed, 9);
  const onset = [],
    report = [];
  const truth = new Array(cutoff + 1).fill(0);
  for (let s = sc.B; s <= sc.B + cutoff; s++) {
    for (let d = 0; d <= L; d++) {
      const n = rpois(sc.C[s] * sc.f[s][d], rng);
      truth[s - sc.B] += n;
      if (s + d - sc.B > cutoff) continue;
      for (let k = 0; k < n; k++) {
        report.push(s + d - sc.B);
        onset.push(rng() < 0.4 ? null : s - sc.B);
      }
    }
  }
  return { onset, report, truth, delay: Array.from(sc.f[sc.B]) };
}

/** Linear-interpolation quantile of sorted values. */
const quantile = (v, p) => {
  const i = (v.length - 1) * p,
    lo = Math.floor(i),
    hi = Math.ceil(i);
  return v[lo] + (v[hi] - v[lo]) * (i - lo);
};

/** Coverage of the 50% and 95% intervals over the last `days` days. */
function coverage(res, truth, days = 28) {
  let n = 0,
    c50 = 0,
    c95 = 0,
    med = 0,
    tot = 0;
  for (let s = truth.length - days; s < truth.length; s++) {
    if (!Number.isFinite(res.point[s])) continue;
    const v = res.samples.map(x => x[s]).sort((a, b) => a - b);
    const t = truth[s];
    n++;
    if (t >= quantile(v, 0.25) && t <= quantile(v, 0.75)) c50++;
    if (t >= quantile(v, 0.025) && t <= quantile(v, 0.975)) c95++;
    med += quantile(v, 0.5);
    tot += t;
  }
  return { c50: c50 / n, c95: c95 / n, rel: med / tot - 1 };
}

for (const [method, given, bootstrap] of [
  ["gen", true, false],
  ["backsampling", true, false],
  ["gen", false, true]
]) {
  test(`${method} nowcast covers the eventual truth (${given ? "true" : "estimated"} delay)`, () => {
    const out = [1, 2, 3].map(seed => {
      const sim = simulate(seed);
      const res = M.nowcastLinelist({
        onset: sim.onset,
        report: sim.report,
        cutoff: CUTOFF,
        method,
        delay: given ? sim.delay : null,
        nSamples: 300,
        seed,
        bootstrap
      });
      return coverage(res, sim.truth);
    });
    const mean = k => out.reduce((a, o) => a + o[k], 0) / out.length;
    assert.ok(mean("c50") >= 0.3 && mean("c50") <= 0.85, `50%: ${mean("c50")}`);
    assert.ok(mean("c95") >= 0.85, `95%: ${mean("c95")}`);
    assert.ok(Math.abs(mean("rel")) < 0.08, `median: ${mean("rel")}`);
  });
}

test("weekly median is close to the truth in the last weeks", () => {
  const sim = simulate(4);
  const res = M.nowcastLinelist({
    onset: sim.onset,
    report: sim.report,
    cutoff: CUTOFF,
    delay: sim.delay,
    nSamples: 300
  });
  const W = res.weekStart.length;
  for (let w = W - 3; w < W; w++) {
    let t = 0;
    for (let d = 0; d < 7; d++) t += sim.truth[res.weekStart[w] + d];
    const v = res.weeklySamples.map(x => x[w]).sort((a, b) => a - b);
    assert.ok(Math.abs(quantile(v, 0.5) / t - 1) < 0.15, `week ${w}`);
  }
});

const small = (() => {
  const sim = simulate(5, 90);
  return { onset: sim.onset, report: sim.report };
})();

test("quantiles are monotone in both tables", () => {
  const res = M.nowcastLinelist({ ...small, nSamples: 200 });
  for (const table of [res.dailyQuantiles, res.weeklyQuantiles]) {
    assert.ok(table.length > 0);
    const byDate = new Map();
    for (const r of table) {
      if (!byDate.has(r.target_end_date)) byDate.set(r.target_end_date, []);
      byDate.get(r.target_end_date).push(r);
    }
    for (const rows of byDate.values()) {
      assert.deepEqual(
        rows.map(r => r.quantile),
        Array.from(M.QUANTILES)
      );
      for (let k = 1; k < rows.length; k++)
        assert.ok(rows[k].value >= rows[k - 1].value);
    }
  }
});

test("output is reproducible with a seed", () => {
  const a = M.nowcastLinelist({ ...small, nSamples: 30, seed: 7 });
  const b = M.nowcastLinelist({ ...small, nSamples: 30, seed: 7 });
  const c = M.nowcastLinelist({ ...small, nSamples: 30, seed: 8 });
  assert.deepEqual(a.samples, b.samples);
  assert.deepEqual(a.weeklyQuantiles, b.weeklyQuantiles);
  assert.notDeepEqual(a.samples, c.samples);
});

for (const method of M.NOWCAST_METHODS) {
  test(`${method}: point and samples respect the truncation correction`, () => {
    const delay = Array.from(M.delayPmf(8, 1)).slice(0, L + 1);
    const res = M.nowcastLinelist({ ...small, method, delay, nSamples: 20 });
    const imp = M.imputeLinelist({ ...small, delay, maxDelay: L });
    const Fc = [];
    let acc = 0;
    for (let d = 0; d <= L; d++) Fc.push((acc += res.delay[d]));
    res.reported.forEach((F, s) => {
      assert.ok(Math.abs(F - Fc[Math.min(res.cutoff - s, L)]) < 1e-12);
      const ok = F >= 0.05;
      assert.equal(Number.isFinite(res.point[s]), ok);
      if (ok && method !== "gen") {
        const want = imp.imputed[method][s] / F;
        assert.ok(Math.abs(res.point[s] - want) < 1e-9 * Math.max(1, want));
      }
      for (const x of res.samples) {
        assert.equal(Number.isFinite(x[s]), ok);
        if (ok) {
          assert.ok(Number.isInteger(x[s]));
          assert.ok(x[s] >= imp.complete[s]);
        }
      }
    });
  });
}

test("days under minReported have no estimate, nor do their weeks", () => {
  const res = M.nowcastLinelist({ ...small, nSamples: 20, minReported: 0.3 });
  const none = res.reported.map(F => F < 0.3);
  assert.ok(none.some(Boolean));
  const days = new Set(res.dailyQuantiles.map(r => r.target_end_date));
  none.forEach((n, s) => assert.equal(days.has(s), !n));
  assert.ok(Number.isNaN(res.pointWeekly.at(-1)));
  const last = res.weekStart.at(-1) + 6;
  assert.ok(res.weeklyQuantiles.every(r => r.target_end_date !== last));
});

test("weeks end on the cut-off and dates follow the origin", () => {
  const res = M.nowcastLinelist({
    ...small,
    nSamples: 20,
    minReported: 0.01,
    origin: "2026-05-08"
  });
  assert.equal(res.cutoff, 90);
  assert.equal(res.weekStart.at(-1) + 6, res.cutoff);
  const row = res.weeklyQuantiles.at(-1);
  assert.equal(row.week_start, "2026-07-31"); // day 84
  assert.equal(row.target_end_date, "2026-08-06"); // day 90
  assert.equal(res.dailyQuantiles[0].target_end_date, "2026-05-08");
  const s0 = res.weekStart[0];
  const x = res.samples[0];
  const sum = x.slice(s0, s0 + 7).reduce((a, b) => a + b, 0);
  assert.equal(res.weeklySamples[0][0], sum);
});

test("input validation", () => {
  const base = { onset: [0, 1, null, 3], report: [2, 3, 4, 5] };
  const bad = [
    [{ method: "unknown" }, /method must be one of/],
    [{ method: "em" }, /method must be one of/],
    [{ nSamples: 0 }, /nSamples/],
    [{ nSamples: 2.5 }, /nSamples/],
    [{ iterations: 0 }, /iterations/],
    [{ maxDelay: -1 }, /maxDelay/],
    [{ seed: 1.5 }, /seed/],
    [{ minReported: 0 }, /minReported/],
    [{ minReported: 1.5 }, /minReported/],
    [{ quantiles: [] }, /quantiles/],
    [{ quantiles: [0.5, 1.2] }, /quantiles/],
    [{ delay: [0, 0] }, /positive mass/],
    [{ delay: [0.5, -0.1] }, /non-negative/],
    [{ cutoff: -1 }, /cutoff/],
    [{ origin: "not a date" }, /origin/]
  ];
  for (const [opts, re] of bad)
    assert.throws(() => M.nowcastLinelist({ ...base, ...opts }), re);
  assert.throws(
    () => M.nowcastLinelist({ onset: [1], report: [1, 2] }),
    /same length/
  );
  assert.throws(
    () => M.nowcastLinelist({ onset: [null], report: [null] }),
    /no valid report days/
  );
  assert.throws(
    () => M.nowcastLinelist({ onset: [null, null], report: [1, 2] }),
    /no complete cases/
  );
  const res = M.nowcastLinelist({
    onset: [0, 5, null, 1, 1.5],
    report: [2, 4, 3, 3, 3],
    nSamples: 5
  });
  assert.equal(res.dropped, 2); // onset after report, non-integer onset
});
