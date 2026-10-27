// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Delays that change over time: per-period EM delays and delay matrices.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { M } from "./helpers.mjs";

const FIXTURE = JSON.parse(
  readFileSync(
    new URL("./fixtures/cross-language.json", import.meta.url),
    "utf8"
  )
);

/**
 * A line list of rounded expected counts with a mean delay falling from 12
 * to 5 days and steady growth, cut at `cutoff`. Every 5th and 5th + 1 case
 * misses its onset. Returns the onsets of cases reported by the cut-off,
 * the eventual onsets and the true delay of each onset day.
 */
function falling(cutoff = 120) {
  const sc = M.scenario({
    ...M.DEFAULTS,
    knots: M.KNOTS.map(() => 0.03),
    mu0: 12,
    mu1: 5,
    cv: 0.8,
    O0: 100
  });
  const onset = [],
    report = [];
  const reported = new Array(cutoff + 1).fill(0);
  const eventual = new Array(cutoff + 1).fill(0);
  let i = 0;
  for (let s = sc.B; s <= sc.B + cutoff; s++) {
    for (let d = 0; d <= M.D; d++) {
      const n = Math.round(sc.C[s] * sc.f[s][d]);
      eventual[s - sc.B] += n;
      if (s + d - sc.B > cutoff) continue;
      reported[s - sc.B] += n;
      for (let k = 0; k < n; k++, i++) {
        report.push(s + d - sc.B);
        onset.push(i % 5 < 2 ? null : s - sc.B);
      }
    }
  }
  const delay = Array.from({ length: cutoff + 1 }, (_, s) =>
    Array.from(sc.f[s + sc.B]).slice(0, M.D + 1)
  );
  return { onset, report, cutoff, reported, eventual, delay };
}

/** Full weeks ending on the last day. */
const weeks = x => {
  const n = Math.floor(x.length / 7),
    off = x.length - 7 * n;
  return Array.from({ length: n }, (_, w) =>
    x.slice(off + 7 * w, off + 7 * w + 7).reduce((a, b) => a + b, 0)
  );
};
const meanAbsRel = (a, b) =>
  a.reduce((t, v, i) => t + Math.abs(v / b[i] - 1), 0) / a.length;

test("delayPeriods counts periods back from the cut-off", () => {
  assert.deepEqual(M.delayPeriods(9), new Array(10).fill(0));
  assert.deepEqual(M.delayPeriods(9, 4), [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
  assert.deepEqual(M.delayPeriods(9, 10), new Array(10).fill(0));
  for (const bad of [0, -1, 2.5])
    assert.throws(() => M.delayPeriods(9, bad), /delayPeriod/);
});

test("per-period EM beats static EM when delays fall", () => {
  const ll = falling();
  const base = { onset: ll.onset, report: ll.report, cutoff: ll.cutoff };
  const err = res => {
    const a = weeks(res.imputed.gen),
      b = weeks(ll.reported);
    return a.reduce((t, v, i) => t + Math.abs(v - b[i]), 0);
  };
  const stat = M.imputeLinelist(base);
  const per = M.imputeLinelist({ ...base, delayPeriod: 14 });
  assert.ok(err(per) < err(stat), `${err(per)} vs ${err(stat)}`);
  const now = res =>
    meanAbsRel(weeks(res.point).slice(-4), weeks(ll.eventual).slice(-4));
  const opts = { ...base, nSamples: 1, bootstrap: false, minReported: 1e-3 };
  const ns = M.nowcastLinelist(opts);
  const np = M.nowcastLinelist({ ...opts, delayPeriod: 14 });
  assert.ok(now(np) < 0.75 * now(ns), `${now(np)} vs ${now(ns)}`);
  // the per-period delays shorten over time
  const mean = row => row.reduce((t, p, d) => t + p * d, 0);
  const means = per.generativeDelayByPeriod.map(mean);
  assert.ok(means[0] > means[means.length - 1] + 2);
  assert.equal(per.delayPeriodStart.length, 9);
});

test("the default is one static delay and one period matches it", () => {
  const ll = falling(80);
  const base = { onset: ll.onset, report: ll.report, cutoff: ll.cutoff };
  const res = M.imputeLinelist(base);
  assert.deepEqual(res.generativeDelayByPeriod, [res.generativeDelay]);
  assert.deepEqual(res.delayPeriodStart, [0]);
  const one = M.imputeLinelist({ ...base, delayPeriod: 81 });
  assert.deepEqual(one.imputed.gen, res.imputed.gen);
  const nc = M.nowcastLinelist({ ...base, nSamples: 10 });
  const nc1 = M.nowcastLinelist({ ...base, nSamples: 10, delayPeriod: 500 });
  assert.deepEqual(nc1.samples, nc.samples);
});

test("a period with no cases takes the previous period's delay", () => {
  const onset = [],
    report = [];
  for (let s = 0; s < 42; s++) {
    if (s >= 14 && s < 28) continue;
    const ds = s < 14 ? [0, 1, 2, 3, 5] : [2, 4, 6, 7, 9];
    for (const d of ds) {
      onset.push(s);
      report.push(s + d);
    }
  }
  const res = M.imputeLinelist({ onset, report, cutoff: 55, delayPeriod: 14 });
  const f = res.generativeDelayByPeriod;
  assert.deepEqual(res.delayPeriodStart, [0, 14, 28, 42]);
  const close = (a, b, tol) => a.every((v, d) => Math.abs(v - b[d]) < tol);
  assert.ok(close(f[1], f[0], 1e-9));
  assert.ok(close(f[3], f[2], 1e-9));
  assert.ok(!close(f[2], f[0], 1e-3));
});

test("a short line list works with periods", () => {
  const input = {
    onset: [0, 2, null, 5, null],
    report: [3, 6, 7, 9, 12],
    cutoff: 12,
    delayPeriod: 4
  };
  const res = M.imputeLinelist(input);
  assert.equal(res.generativeDelayByPeriod.length, 4);
  const total = res.imputed.gen.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(total - 5) < 1e-9);
  const nc = M.nowcastLinelist({ ...input, nSamples: 30 });
  assert.equal(nc.samples.length, 30);
  assert.equal(nc.delayByPeriod.length, 4);
  assert.deepEqual(nc.periodStart, [0, 1, 5, 9]);
});

test("a delay matrix is a plug-in and beats a static plug-in", () => {
  const ll = falling();
  const base = { onset: ll.onset, report: ll.report, cutoff: ll.cutoff };
  const res = M.imputeLinelist({ ...base, delay: ll.delay });
  assert.equal(res.generativeDelayByPeriod.length, ll.cutoff + 1);
  const row = ll.delay[7],
    rowSum = row.reduce((a, b) => a + b, 0);
  res.generativeDelayByPeriod[7].forEach((v, d) =>
    assert.ok(Math.abs(v - row[d] / rowSum) < 1e-12)
  );
  const nMiss = res.missingByReport.reduce((a, b) => a + b, 0);
  const fwd = res.imputed.fwd.reduce((a, b) => a + b, 0);
  const comp = res.complete.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(fwd - comp - nMiss) < 1e-6);
  const opts = { ...base, nSamples: 1, bootstrap: false, minReported: 1e-3 };
  const now = d => {
    const nc = M.nowcastLinelist({ ...opts, delay: d });
    return meanAbsRel(weeks(nc.point).slice(-4), weeks(ll.eventual).slice(-4));
  };
  const staticDelay = Array.from(M.delayPmf(8, 0.8)).slice(0, M.D + 1);
  assert.ok(now(ll.delay) < 0.5 * now(staticDelay));
  assert.throws(
    () => M.imputeLinelist({ ...base, delay: ll.delay.slice(1) }),
    /one row per onset day/
  );
  assert.throws(
    () => M.nowcastLinelist({ ...base, delay: ll.delay[0], delayPeriod: 7 }),
    /not both/
  );
  assert.throws(
    () => M.imputeLinelist({ ...base, delayShrinkage: 0 }),
    /delayShrinkage/
  );
});

test("every nowcast method runs with periods", () => {
  const ll = falling(70);
  for (const method of M.NOWCAST_METHODS) {
    const res = M.nowcastLinelist({
      onset: ll.onset,
      report: ll.report,
      cutoff: ll.cutoff,
      method,
      delayPeriod: 21,
      nSamples: 10
    });
    assert.deepEqual(res.periodStart, [0, 8, 29, 50]);
    for (let s = 0; s < 60; s++) {
      assert.ok(Number.isFinite(res.point[s]), `${method} day ${s}`);
      assert.ok(res.samples.every(x => Number.isFinite(x[s])));
    }
  }
});

/** Largest relative error over weeks with at least 50 cases. */
function weeklyRelErr(a, b) {
  const wa = weeks(a),
    wb = weeks(b);
  let m = 0;
  wb.forEach((v, i) => {
    if (v >= 50) m = Math.max(m, Math.abs(wa[i] / v - 1));
  });
  return m;
}

test("per-period EM matches the shared fixture", () => {
  const fx = FIXTURE.linelist_periods;
  const res = M.imputeLinelist({
    onset: fx.onset,
    report: fx.report,
    cutoff: fx.cutoff,
    maxDelay: fx.max_delay,
    delayPeriod: fx.delay_period,
    delayShrinkage: fx.delay_shrinkage
  });
  assert.deepEqual(res.delayPeriodStart, fx.delay_period_start);
  for (const k of ["backsampling", "mean"])
    res.imputed[k].forEach((v, i) =>
      assert.ok(Math.abs(v - fx.imputed[k][i]) < 1e-9, `${k} day ${i}`)
    );
  assert.ok(weeklyRelErr(res.imputed.gen, fx.imputed.em) <= 0.02);
});

test("a supplied delay matrix matches the shared fixture", () => {
  const fx = FIXTURE.linelist_delay_matrix;
  const res = M.imputeLinelist({
    onset: fx.onset,
    report: fx.report,
    cutoff: fx.cutoff,
    maxDelay: fx.max_delay,
    delay: fx.delay
  });
  for (const [k, ref] of [
    ["backsampling", "backsampling"],
    ["mean", "mean"],
    ["fwd", "forward"]
  ])
    res.imputed[k].forEach((v, i) =>
      assert.ok(Math.abs(v - fx.imputed[ref][i]) < 1e-9, `${k} day ${i}`)
    );
  assert.ok(weeklyRelErr(res.imputed.gen, fx.imputed.em) <= 0.02);
});
