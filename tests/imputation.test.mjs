// Tests of the maths in imputation.js, run with `node --test "tests/*.test.mjs"`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const M = createRequire(import.meta.url)("../imputation.js");

function run(set) {
  const p = { ...M.DEFAULTS, ...set };
  const sc = M.scenario(p);
  const ex = M.expectedData(sc);
  const R = M.runMethods(sc, ex.comp, ex.miss, ex.empCounts);
  const W = Math.floor(Math.min(sc.N, sc.Td + 1) / 7);
  const tw = M.weekly(ex.truth, W, sc.B);
  const mw = {};
  for (const m of M.METHODS) mw[m.key] = M.weekly(R[m.key], W, sc.B);
  return { p, sc, ex, R, W, tw, mw };
}
const maxRelErr = (mw, tw) => Math.max(...Array.from(mw, (v, w) =>
  tw[w] >= 10 ? Math.abs(v / tw[w] - 1) : 0));
const steady = r => ({ knots: M.KNOTS.map(() => r) });
const bvd = M.PRESETS[0].set;

test("rounded Gamma delay has the requested mean", () => {
  for (const mu of [4, 8, 14]) {
    assert.ok(Math.abs(M.pmfMean(M.delayPmf(mu, 1)) - mu) < 0.1, `mu ${mu}`);
  }
});

test("pooled empirical delay matches the backward delay in steady growth", () => {
  const r = 0.05;
  const { R } = run({ ...steady(r), retro: false, cutoff: 150 });
  const f = M.delayPmf(8, 1);
  let num = 0, den = 0;
  for (let d = 0; d < f.length; d++) {
    const w = f[d] * Math.exp(-r * d);
    num += d * w; den += w;
  }
  assert.ok(Math.abs(M.pmfMean(R.g) - num / den) < 0.05);
});

test("hub method is unbiased in steady growth, real time", () => {
  const { mw, tw } = run({ ...steady(0.05), retro: false, cutoff: 150 });
  assert.ok(maxRelErr(mw.hub, tw) < 0.005);
});

test("forward draw overstates counts in steady growth", () => {
  const { mw, tw } = run({ ...steady(0.05), retro: false, cutoff: 150 });
  assert.ok(maxRelErr(mw.fwd, tw) > 0.05);
});

test("hub method differs from truth when growth changes", () => {
  const { mw, tw } = run(M.PRESETS.find(p => p.name === "Sharp turnaround").set);
  assert.ok(maxRelErr(mw.hub, tw) > 0.05);
});

test("EM back-projection recovers the truth when missing at random", () => {
  for (const set of [bvd, { ...bvd, retro: false, cutoff: 120 },
    { ...bvd, mu0: 12, mu1: 4 }, { ...bvd, pi0: 0.2, pi1: 0.6 }]) {
    const { mw, tw } = run(set);
    assert.ok(maxRelErr(mw.gen, tw) < 1e-6);
  }
});

test("growth rate error of EM is near zero", () => {
  const { mw, tw } = run(M.PRESETS.find(p => p.name === "Two waves").set);
  const gT = M.growth(tw), gE = M.growth(mw.gen);
  for (let w = 0; w < gT.length; w++) {
    if (gT[w] !== null && tw[w] >= 10) assert.ok(Math.abs(gE[w] - gT[w]) < 1e-6);
  }
});

test("expected hub error lies inside the stochastic 90% band", () => {
  const { sc, W, mw, tw } = run(bvd);
  const st = M.stochastic(sc, W, 60);
  let inside = 0, n = 0;
  for (const b of st.bands.hub) {
    const w = (b.x - 3.5) / 7;
    if (tw[w] < 50) continue;
    const e = mw.hub[w] / tw[w] - 1;
    n++;
    if (e >= b.lo - 0.01 && e <= b.hi + 0.01) inside++;
  }
  assert.ok(n > 5 && inside / n > 0.8, `${inside}/${n}`);
});

test("run() matches the lower-level pipeline", () => {
  const res = M.run(bvd);
  const { mw, tw } = run(bvd);
  assert.deepEqual(res.truth, Array.from(tw));
  assert.deepEqual(res.imputed.hub, Array.from(mw.hub));
  assert.ok(Math.abs(res.metrics.gen.maxRel) < 1e-6);
});

test("Rt is 1 at zero growth and above 1 in growth", () => {
  assert.ok(Math.abs(M.rToR(0) - 1) < 1e-12);
  assert.ok(M.rToR(0.05) > 1 && M.rToR(-0.05) < 1);
});

test("every preset runs without non-finite values", () => {
  for (const pr of M.PRESETS) {
    const { mw } = run(pr.set);
    for (const m of M.METHODS) {
      assert.ok(Array.from(mw[m.key]).every(Number.isFinite), pr.name);
    }
  }
});
