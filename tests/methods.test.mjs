// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Scenarios, the four imputation methods and run() in epidemic-dates.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  M,
  LONG,
  bvd,
  maxRelErr,
  pipeline,
  preset,
  steady,
  sum
} from "./helpers.mjs";

// use mu0, mu1, pi0 and pi1 in place of the baseline curves
const linear = { muKnots: null, piKnots: null };

test("hub method is unbiased in steady growth, real time", () => {
  const { mw, tw } = pipeline({ ...steady(0.05), retro: false, cutoff: 150 });
  assert.ok(maxRelErr(mw.hub, tw) < 0.005);
});

test("forward draw overstates counts in steady growth", () => {
  const { mw, tw } = pipeline({ ...steady(0.05), retro: false, cutoff: 150 });
  assert.ok(maxRelErr(mw.fwd, tw) > 0.05);
});

test("hub method differs from truth when growth changes", () => {
  const { mw, tw } = pipeline({ ...preset("Sharp turnaround"), ...LONG });
  assert.ok(maxRelErr(mw.hub, tw) > 0.05);
});

test("EM back-projection recovers the truth when missing at random", () => {
  for (const set of [
    bvd,
    { ...bvd, retro: false, cutoff: 120 },
    { ...bvd, ...linear, mu0: 12, mu1: 4 },
    { ...bvd, ...linear, pi0: 0.2, pi1: 0.6 }
  ]) {
    const { mw, tw } = pipeline(set);
    assert.ok(maxRelErr(mw.gen, tw) < 1e-6);
  }
});

test("growth rate error of EM is near zero", () => {
  const { mw, tw } = pipeline(preset("Two waves"));
  const gT = M.growth(tw),
    gE = M.growth(mw.gen);
  for (let w = 0; w < gT.length; w++) {
    if (gT[w] !== null && tw[w] >= 10) {
      assert.ok(Math.abs(gE[w] - gT[w]) < 1e-6);
    }
  }
});

test("every method keeps the total case count", () => {
  // retrospective, so no imputed onset falls after the last day
  const { ex, R } = pipeline({ ...bvd, retro: true });
  const total = sum(ex.comp) + sum(ex.miss);
  for (const m of M.METHODS) {
    const lost = total - sum(R[m.key]);
    assert.ok(lost >= -1e-6 && lost / total < 1e-3, m.key);
  }
});

test("convImpute and shiftImpute drop onsets before day 0 and past S", () => {
  const comp = [0, 0, 0],
    miss = [2, 0, 4];
  // delay 0 or 1 with equal probability
  assert.deepEqual(
    Array.from(M.convImpute(comp, miss, [0.5, 0.5], 3)),
    [1, 2, 2]
  );
  assert.deepEqual(Array.from(M.shiftImpute(comp, miss, 2, 3)), [4, 0, 0]);
  // onsets at day S or later are dropped; the output keeps the length of comp
  assert.deepEqual(Array.from(M.shiftImpute(comp, miss, 0, 2)), [2, 0, 0]);
});

test("emImpute calls back on each iteration", () => {
  const { sc, ex } = pipeline(bvd);
  const seen = [];
  const out = M.emImpute(ex.comp, ex.miss, sc, ex.comp, 5, (it, cur, lam) => {
    seen.push(it);
    assert.equal(cur.length, sc.S);
    assert.equal(lam.length, sc.S);
  });
  assert.deepEqual(seen, [1, 2, 3, 4, 5]);
  assert.ok(out.every(Number.isFinite));
});

test("missingness linked to delay is scaled by the onset day's mean delay", () => {
  const set = {
    ...bvd,
    ...linear,
    mu0: 12,
    mu1: 6,
    pi0: 0.3,
    pi1: 0.3,
    beta: 1.5
  };
  const { sc } = pipeline(set);
  // onset on the last day shown has mean delay mu1 = 6, so a 6-day delay
  // is the reference and gets the base probability
  const s = sc.B + sc.N - 1;
  assert.ok(Math.abs(sc.pim(s + 6, 6) - 0.3) < 1e-12);
  // onset on day 0 has mean delay mu0 = 12
  assert.ok(Math.abs(sc.pim(sc.B + 12, 12) - 0.3) < 1e-12);
  // longer delays than the mean are more likely to be missing
  assert.ok(sc.pim(s + 12, 12) > 0.3 && sc.pim(s + 2, 2) < 0.3);
});

test("beta of 0 keeps missingness at random given report day", () => {
  const { sc } = pipeline({ ...bvd, ...linear, pi0: 0.1, pi1: 0.5 });
  const i = sc.B + 50;
  assert.equal(sc.pim(i, 1), sc.pim(i, 20));
  assert.equal(sc.pim(sc.B, 0), 0.1);
  assert.equal(sc.pim(sc.B + sc.N - 1, 0), 0.5);
});

test("scenario cut-off, retrospective view and an ended outbreak", () => {
  const rt = pipeline({ ...bvd, retro: false, cutoff: 100 }).sc;
  assert.equal(rt.Td, 100);
  const late = pipeline({ ...bvd, retro: false, cutoff: 1000 }).sc;
  assert.equal(late.Td, late.N - 1);
  const retro = pipeline({ ...bvd, retro: true, ended: true }).sc;
  assert.equal(retro.Td, retro.N - 1 + M.D);
  assert.equal(retro.C[retro.B + retro.N], 0);
  assert.ok(retro.C[retro.B + retro.N - 1] > 0);
});

test("run() matches the lower-level pipeline", () => {
  const res = M.run(bvd);
  const { p, sc, ex, R, W } = pipeline(bvd);
  const cs = M.comparisonSeries(p, sc, ex, R);
  const wk = x => Array.from(M.weekly(x, W, sc.B + sc.wk0));
  assert.deepEqual(res.truth, wk(cs.truth));
  assert.deepEqual(res.imputed.hub, wk(cs.methods.hub));
  assert.ok(Math.abs(res.metrics.gen.maxRel) < 1e-6);
});

test("run() returns one entry per week and method", () => {
  const res = M.run({ ...bvd, retro: false, cutoff: 120 });
  const W = res.truth.length;
  assert.equal(res.weekStart.length, W);
  assert.equal(res.complete.length, W);
  assert.equal(res.pooledDelay.length, M.D + 1);
  assert.equal(res.forwardDelay.length, M.D + 1);
  for (const m of M.METHODS) {
    assert.equal(res.imputed[m.key].length, W);
    assert.equal(res.rt[m.key].length, W);
    const met = res.metrics[m.key];
    assert.equal(met.lastWeeks.length, 3);
    assert.ok(met.lastWeeks.every(Number.isFinite));
    const big = Math.max(...met.lastWeeks.map(Math.abs));
    assert.equal(Math.abs(met.lastMaxAbs), big);
    assert.ok(met.lastWeeks.includes(met.lastMaxAbs));
  }
  const retro = M.run({ ...bvd, retro: true }).metrics.hub;
  assert.deepEqual(retro.lastWeeks, []);
  assert.equal(retro.lastMaxAbs, null);
});

test("every preset runs without non-finite values", () => {
  for (const pr of M.PRESETS) {
    const { mw } = pipeline(pr.set);
    for (const m of M.METHODS) {
      assert.ok(Array.from(mw[m.key]).every(Number.isFinite), pr.name);
    }
  }
});

test("nowcast: corrected generative method recovers all eventual onsets", () => {
  const res = M.run({ ...bvd, retro: false, nowcast: true, cutoff: 150 });
  res.truth.forEach((t, w) => {
    if (t >= 10 && res.complete[w]) {
      assert.ok(Math.abs(res.imputed.gen[w] / t - 1) < 1e-6);
    }
  });
});

test("nowcast: hub method is unbiased in steady growth", () => {
  const res = M.run({
    ...steady(0.05),
    retro: false,
    nowcast: true,
    cutoff: 150
  });
  res.truth.forEach((t, w) => {
    if (t >= 10 && res.complete[w]) {
      assert.ok(Math.abs(res.imputed.hub[w] / t - 1) < 0.005);
    }
  });
});

test("correctTruncation zeroes days with too little reported", () => {
  const sc = { cover: Float64Array.from([1, 0.5, 0.01]) };
  assert.deepEqual(Array.from(M.correctTruncation([2, 2, 2], sc)), [2, 4, 0]);
  assert.deepEqual(
    Array.from(M.correctTruncation([2, 2, 2], sc, 0.6)),
    [2, 0, 0]
  );
});

test("metrics on a known peak and turning point", () => {
  const tw = [10, 20, 40, 20, 10, 10];
  const mw = [10, 20, 30, 30, 10, 10];
  const ok = tw.map(() => true);
  const met = M.metrics(tw, mw, M.growth(tw), M.growth(mw), ok, 6, true);
  assert.equal(met.maxRel, 0.5);
  assert.equal(met.maxRelW, 3);
  assert.equal(met.peakShift, 0);
  assert.ok(Math.abs(met.peakHeight + 0.25) < 1e-12);
  assert.deepEqual(met.lastWeeks, []);
  assert.equal(met.lastMaxAbs, null);
  assert.ok(met.zT !== null && met.zM !== null);
  assert.equal(met.lag, met.zM - met.zT);
  // too few cases: no weekly error, no peak
  const flat = M.metrics([1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0], ok, 3);
  assert.equal(flat.maxRel, 0);
  assert.equal(flat.maxRelW, -1);
  assert.equal(flat.peakShift, null);
});

test("exported constants are frozen", () => {
  assert.ok(Object.isFrozen(M));
  assert.ok(Object.isFrozen(M.DEFAULTS) && Object.isFrozen(M.DEFAULTS.knots));
  assert.ok(Object.isFrozen(M.PRESETS[0].set.asc));
  assert.ok(Object.isFrozen(M.KNOTS) && Object.isFrozen(M.METHODS[0]));
  assert.throws(() => {
    M.DEFAULTS.days = 1;
  }, TypeError);
  assert.throws(() => {
    M.KNOTS[0] = 99;
  }, TypeError);
  // a copy can be changed
  const k0 = M.DEFAULTS.knots[0];
  const p = { ...M.DEFAULTS, knots: M.DEFAULTS.knots.slice() };
  p.knots[0] = 1;
  assert.equal(M.DEFAULTS.knots[0], k0);
});
