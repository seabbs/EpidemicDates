// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Seeded stochastic replicates in epidemic-dates.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import { M, bvd, pipeline } from "./helpers.mjs";

test("expected hub error lies inside the stochastic 90% band", () => {
  const { sc, W, mw, tw } = pipeline(bvd);
  const st = M.stochastic(sc, W, 60);
  let inside = 0,
    n = 0;
  for (const b of st.bands.hub) {
    const w = (b.x - 3.5 - sc.wk0) / 7;
    if (tw[w] < 50) continue;
    const e = mw.hub[w] / tw[w] - 1;
    n++;
    if (e >= b.lo - 0.01 && e <= b.hi + 0.01) inside++;
  }
  assert.ok(n > 5 && inside / n > 0.8, `${inside}/${n}`);
});

test("stochastic replicates are reproducible with the same seed", () => {
  const { sc, W } = pipeline(bvd);
  const a = M.stochastic(sc, W, 10, false, 7);
  const b = M.stochastic(sc, W, 10, false, 7);
  assert.deepEqual(a, b);
  const c = M.stochastic(sc, W, 10, false, 8);
  assert.notDeepEqual(a.bands.hub, c.bands.hub);
});

test("stochastic replicates default to the scenario seed", () => {
  const { sc, W } = pipeline({ ...bvd, seed: 5 });
  assert.equal(sc.seed, 5);
  assert.deepEqual(
    M.stochastic(sc, W, 10, true),
    M.stochastic(sc, W, 10, true, 5)
  );
});

test("stochastic replicates do not use Math.random", t => {
  const { sc, W } = pipeline(bvd);
  t.mock.method(Math, "random", () => {
    throw new Error("Math.random was called");
  });
  assert.doesNotThrow(() => M.stochastic(sc, W, 3, true));
});

test("stochastic bands are ordered and keyed by method", () => {
  const { sc, W } = pipeline({ ...bvd, retro: false, cutoff: 120 });
  const st = M.stochastic(sc, W, 12, true);
  for (const m of M.METHODS) {
    for (const key of ["bands", "gbands", "ebands"]) {
      assert.ok(Array.isArray(st[key][m.key]), `${key} ${m.key}`);
      for (const b of st[key][m.key]) assert.ok(b.lo <= b.hi);
    }
  }
  assert.ok(st.truthBand.length > 0);
  // too few replicates for any band
  const none = M.stochastic(sc, W, 2);
  assert.equal(none.bands.hub.length, 0);
});
