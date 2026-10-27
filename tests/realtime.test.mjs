// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Knots, time-varying delay and missingness, the mean-shift mask and the
// real-time metrics in epidemic-dates.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import { M, bvd, pipeline } from "./helpers.mjs";

test("KNOTS run every 15 days to day 360", () => {
  assert.equal(M.KNOTS.length, 25);
  assert.equal(M.KNOTS[0], 0);
  assert.equal(M.KNOTS[24], 360);
  assert.ok(M.KNOTS.every((k, j) => k === 15 * j));
  for (const pr of M.PRESETS)
    for (const k of ["knots", "asc", "muKnots", "piKnots"])
      if (pr.set[k]) assert.equal(pr.set[k].length, 25, `${pr.name} ${k}`);
});

test("the default window is 175 days and presets keep their own", () => {
  assert.equal(M.DEFAULTS.days, 175);
  assert.equal(M.run(bvd).params.days, 175);
  const two = M.PRESETS.find(p => p.name === "Two waves").set;
  assert.equal(M.run(two).params.days, 180);
  // a window up to the editor range runs
  const long = M.run({ ...bvd, days: 357 });
  assert.equal(long.truth.length, 51);
});

test("the baseline has a sharp change point near the end", () => {
  const k = M.DEFAULTS.knots;
  const at = d => M.curveAt(k, d);
  assert.ok(at(0) > 0.1);
  assert.ok(Math.abs(at(100)) < 0.005);
  assert.ok(at(150) < -0.01);
  assert.ok(at(168) >= 0.029 && at(175) >= 0.029);
});

test("muKnots and piKnots set the delay and missingness by day", () => {
  const muKnots = M.knotsFrom([
    [0, 12],
    [60, 6]
  ]);
  const piKnots = M.knotsFrom([
    [0, 0.05],
    [60, 0.3]
  ]);
  const { sc } = pipeline({ ...bvd, muKnots, piKnots, beta: 0 });
  // mean delay by onset day
  const mean = i => M.pmfMean(sc.f[sc.B + i]);
  assert.ok(Math.abs(mean(0) - 12) < 0.2);
  assert.ok(Math.abs(mean(30) - 9) < 0.2);
  assert.ok(Math.abs(mean(100) - 6) < 0.2);
  // missingness by report day
  assert.ok(Math.abs(sc.pim(sc.B, 3) - 0.05) < 1e-12);
  assert.ok(Math.abs(sc.pim(sc.B + 30, 3) - 0.175) < 1e-12);
  assert.ok(Math.abs(sc.pim(sc.B + 120, 3) - 0.3) < 1e-12);
  // null knots fall back to the linear mu0/mu1 and pi0/pi1
  const lin = pipeline({
    ...bvd,
    muKnots: null,
    piKnots: null,
    mu0: 5,
    mu1: 5,
    pi0: 0.2,
    pi1: 0.2
  }).sc;
  assert.ok(Math.abs(M.pmfMean(lin.f[lin.B + 50]) - 5) < 0.2);
  assert.equal(lin.pim(lin.B + 50, 2), 0.2);
});

test("EM back-projection recovers the truth with the baseline curves", () => {
  const res = M.run(bvd);
  res.truth.forEach((t, w) => {
    if (t >= 10 && res.complete[w])
      assert.ok(Math.abs(res.imputed.gen[w] / t - 1) < 1e-6, `week ${w}`);
  });
});

test("maskWeeks blanks weeks with days after the last day", () => {
  const sc = { wk0: 2 };
  // weeks cover days 2-8, 9-15, 16-22
  const out = M.maskWeeks([1, 2, 3], sc, 15);
  assert.deepEqual(Array.from(out.slice(0, 2)), [1, 2]);
  assert.ok(Number.isNaN(out[2]));
  assert.ok(Number.isNaN(M.maskWeeks([1, 2, 3], sc, 14)[1]));
  assert.deepEqual(Array.from(M.maskWeeks([1, 2], sc, null)), [1, 2]);
});

test("comparisonSeries stops the mean shift at T - m in real time", () => {
  const { p, sc, ex, R } = pipeline({ ...bvd, cutoff: 150 });
  const cs = M.comparisonSeries(p, sc, ex, R);
  assert.equal(cs.m, R.m);
  assert.equal(cs.lastDay.mean, 150 - R.m);
  for (const k of ["backsampling", "fwd", "gen"]) assert.equal(cs.lastDay[k], null);
  const retro = pipeline({ ...bvd, retro: true });
  const cr = M.comparisonSeries(retro.p, retro.sc, retro.ex, retro.R);
  assert.equal(cr.lastDay.mean, null);
});

test("run() leaves the mean shift's last weeks out in real time", () => {
  const res = M.run({ ...bvd, cutoff: 150 });
  const m = Math.round(M.pmfMean(res.pooledDelay));
  const W = res.truth.length;
  res.weekStart.forEach((s, w) => {
    const masked = s + 6 > 150 - m;
    assert.equal(res.imputed.mean[w] === null, masked, `week ${w}`);
    if (masked) assert.equal(res.rt.mean[w], null);
  });
  assert.equal(res.imputed.mean[W - 1], null);
  assert.ok(res.imputed.backsampling.every(Number.isFinite));
  // the metrics skip the masked weeks
  const met = res.metrics.mean;
  assert.ok(Number.isFinite(met.maxRel) && Number.isFinite(met.meanAbs));
  assert.ok(met.maxRelW < W - 1);
  assert.equal(met.lastWeeks.length, 3);
  // retrospective: nothing is masked
  const retro = M.run({ ...bvd, retro: true });
  assert.ok(retro.imputed.mean.every(Number.isFinite));
});

test("metrics list the last 3 weeks one by one", () => {
  const tw = [100, 100, 100, 100, 100];
  const mw = [100, 90, 101.5, 104.7, 88.2];
  const ok = tw.map(() => true);
  const met = M.metrics(tw, mw, M.growth(tw), M.growth(mw), ok, 5, false);
  const r = v => Math.round(v * 1000) / 1000;
  assert.deepEqual(met.lastWeeks.map(r), [0.015, 0.047, -0.118]);
  assert.equal(r(met.lastMaxAbs), -0.118);
  // weeks with no value are skipped
  const nan = M.metrics(
    tw,
    [100, 90, 101.5, 104.7, NaN],
    M.growth(tw),
    M.growth([100, 90, 101.5, 104.7, NaN]),
    ok,
    5,
    false
  );
  assert.deepEqual(nan.lastWeeks.map(r), [-0.1, 0.015, 0.047]);
  assert.equal(r(nan.maxRel), -0.1);
});

test("stochastic bands leave out the mean shift's last week", () => {
  const { sc, W } = pipeline({ ...bvd, cutoff: 150 });
  const st = M.stochastic(sc, W, 6, true, 3);
  const lastX = sc.wk0 + 7 * (W - 1) + 3.5;
  assert.ok(!st.bands.mean.some(b => b.x === lastX));
  assert.ok(st.bands.backsampling.some(b => b.x === lastX));
});
