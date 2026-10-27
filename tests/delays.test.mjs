// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Delays, curves, Rt and small helpers in onset-imputation.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { M, pipeline, steady, sum } from "./helpers.mjs";

test("rounded Gamma delay has the requested mean", () => {
  for (const mu of [4, 8, 14]) {
    assert.ok(Math.abs(M.pmfMean(M.delayPmf(mu, 1)) - mu) < 0.1, `mu ${mu}`);
  }
});

test("delay pmf has D + 1 non-negative entries summing to 1", () => {
  for (const [mu, cv] of [
    [1, 0.3],
    [6.6, 0.92],
    [20, 1.6]
  ]) {
    const f = M.delayPmf(mu, cv);
    assert.equal(f.length, M.D + 1);
    assert.ok(f.every(v => v >= 0 && Number.isFinite(v)));
    assert.ok(Math.abs(sum(f) - 1) < 1e-12);
  }
});

test("delayPmf returns a fresh array, so callers cannot corrupt it", () => {
  const a = M.delayPmf(8, 1);
  const before = Array.from(a);
  a.fill(0);
  assert.deepEqual(Array.from(M.delayPmf(8, 1)), before);
});

test("results do not depend on what ran before", () => {
  const set = { mu0: 12, mu1: 4 };
  // a fresh process has no history
  const code =
    "const M = require('./onset-imputation.js');" +
    `process.stdout.write(JSON.stringify(M.run(${JSON.stringify(set)})));`;
  const fresh = execFileSync(process.execPath, ["-e", code], {
    cwd: new URL("..", import.meta.url),
    encoding: "utf8"
  });
  // nearby mean delays used to share a cached pmf
  for (const pr of M.PRESETS) M.run(pr.set);
  M.run({ mu0: 12.0001, mu1: 4.0001 });
  assert.equal(JSON.stringify(M.run(set)), fresh);
});

test("pooled empirical delay matches the backward delay in steady growth", () => {
  const r = 0.05;
  const { R } = pipeline({ ...steady(r), retro: false, cutoff: 150 });
  const f = M.delayPmf(8, 1);
  let num = 0,
    den = 0;
  for (let d = 0; d < f.length; d++) {
    const w = f[d] * Math.exp(-r * d);
    num += d * w;
    den += w;
  }
  assert.ok(Math.abs(M.pmfMean(R.g) - num / den) < 0.05);
});

test("Rt is 1 at zero growth and above 1 in growth", () => {
  assert.ok(Math.abs(M.rToR(0) - 1) < 1e-12);
  assert.ok(M.rToR(0.05) > 1 && M.rToR(-0.05) < 1);
  assert.equal(M.rToR(null), null);
  // the base is floored, so very fast decline stays finite and positive
  assert.ok(M.rToR(-10) > 0 && Number.isFinite(M.rToR(-10)));
});

test("wrongSideOf1 ignores growth rates near zero", () => {
  assert.equal(M.wrongSideOf1(0.01, -0.01), true);
  assert.equal(M.wrongSideOf1(0.01, 0.02), false);
  assert.equal(M.wrongSideOf1(1e-4, -1e-4), false);
});

test("curveAt interpolates between knots and is flat outside", () => {
  const vals = M.KNOTS.map((_, j) => j * j);
  assert.equal(M.curveAt(vals, -10), 0);
  assert.equal(M.curveAt(vals, 0), 0);
  assert.equal(M.curveAt(vals, 15), 1);
  assert.equal(M.curveAt(vals, 7.5), 0.5);
  assert.equal(M.curveAt(vals, 22.5), 2.5);
  assert.equal(M.curveAt(vals, 240), 256);
  assert.equal(M.curveAt(vals, 1000), 256);
});

test("curveAt agrees with the growth rate in a scenario", () => {
  const knots = M.knotsFrom([
    [0, 0.06],
    [50, -0.03],
    [200, 0.01]
  ]);
  const { sc } = pipeline({ knots, rwSd: 0, retro: true });
  for (let i = 0; i < sc.S; i++) {
    assert.equal(sc.r[i], M.curveAt(knots, i - sc.B), `day ${i - sc.B}`);
  }
});

test("knotsFrom is linear between anchors and flat outside", () => {
  const k = M.knotsFrom([
    [30, 1],
    [60, 2]
  ]);
  assert.equal(k.length, M.KNOTS.length);
  assert.equal(k[0], 1); // day 0, before the first anchor
  assert.equal(k[3], 1.5); // day 45
  assert.equal(k[k.length - 1], 2); // day 240, after the last anchor
});

test("normalise sums to 1 and leaves an all-zero input at zero", () => {
  assert.deepEqual(Array.from(M.normalise([1, 3])), [0.25, 0.75]);
  assert.deepEqual(Array.from(M.normalise([0, 0, 0])), [0, 0, 0]);
  assert.equal(M.normalise([]).length, 0);
});

test("clamp and pmfMean", () => {
  assert.equal(M.clamp(5, 0, 1), 1);
  assert.equal(M.clamp(-5, 0, 1), 0);
  assert.equal(M.clamp(0.5, 0, 1), 0.5);
  assert.equal(M.pmfMean([0, 0.5, 0.5]), 1.5);
  assert.equal(M.pmfMean([]), 0);
});

test("weekly sums days and counts days beyond the array as 0", () => {
  const daily = Array.from({ length: 10 }, () => 1);
  assert.deepEqual(Array.from(M.weekly(daily, 2, 0)), [7, 3]);
  assert.deepEqual(Array.from(M.weekly(daily, 1, 5)), [5]);
});

test("growth handles zeros, one week and an offset", () => {
  assert.deepEqual(M.growth([5]), [null]);
  assert.deepEqual(M.growth([0, 0, 0]), [null, null, null]);
  const g = M.growth([1, Math.exp(7), Math.exp(14)]);
  for (const v of g) assert.ok(Math.abs(v - 1) < 1e-12);
  // the offset guards against log(0)
  assert.ok(M.growth([0, 1], 0.5).every(Number.isFinite));
});

test("makeRng is reproducible, in [0, 1), and streams differ", () => {
  const a = M.makeRng(3),
    b = M.makeRng(3),
    c = M.makeRng(3, 1);
  const xa = Array.from({ length: 1000 }, a);
  const xb = Array.from({ length: 1000 }, b);
  const xc = Array.from({ length: 1000 }, c);
  assert.deepEqual(xa, xb);
  assert.notDeepEqual(xa, xc);
  assert.ok(xa.every(u => u >= 0 && u < 1));
  assert.ok(Math.abs(sum(xa) / 1000 - 0.5) < 0.05);
});
