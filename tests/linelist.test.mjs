// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// imputeLinelist in epidemic-dates.js, for real line lists.
import { test } from "node:test";
import assert from "node:assert/strict";
import { M, byWeek, linelist, sum } from "./helpers.mjs";

const finite = r =>
  ["backsampling", "mean", "gen"].every(k =>
    r.imputed[k].every(Number.isFinite)
  );

test("generative method recovers onsets with an estimated delay", () => {
  const set = {
    knots: M.knotsFrom([
      [0, 0.05],
      [60, 0.05],
      [100, -0.02]
    ]),
    O0: 3
  };
  const ll = linelist(set, 150);
  const res = M.imputeLinelist({
    onset: ll.onset,
    report: ll.report,
    cutoff: 150
  });
  const truth = new Array(res.days).fill(0);
  for (const s of ll.truth) truth[s]++;
  const W = 18,
    tw = byWeek(truth, W),
    gw = byWeek(res.imputed.gen, W);
  for (let w = 2; w < W; w++) {
    if (tw[w] >= 100)
      assert.ok(Math.abs(gw[w] / tw[w] - 1) < 0.05, `week ${w}`);
  }
  const n = ll.report.length;
  assert.ok(Math.abs(sum(res.imputed.gen) - n) < 1e-6 * n);
  assert.equal(res.droppedImputed.gen, 0);
});

test("input checks and dropped cases", () => {
  assert.throws(() => M.imputeLinelist({ onset: [1], report: [1, 2] }));
  assert.throws(
    () => M.imputeLinelist({ onset: [null], report: [null] }),
    /no valid report days/
  );
  const res = M.imputeLinelist({ onset: [0, null, 5], report: [2, 3, 4] });
  assert.equal(res.dropped, 1); // onset after report
  assert.equal(res.missingByReport[3], 1);
});

test("every kind of invalid case is dropped", () => {
  const res = M.imputeLinelist({
    onset: [0, 1.5, -1, 0, 3, null, 0, 2],
    report: [1, 2, 2, 50, 2.5, 4, 3, 7],
    cutoff: 5,
    maxDelay: 2
  });
  // non-integer onset, onset before day 0, report after the cut-off,
  // non-integer report, delay over maxDelay, report after the cut-off
  assert.equal(res.dropped, 6);
  assert.equal(res.days, 6);
  assert.equal(res.cutoff, 5);
  assert.equal(sum(res.complete), 1);
  assert.equal(sum(res.missingByReport), 1);
});

test("short maxDelay, short delay and null reports stay finite", () => {
  const onset = [0, 1, null, 3, null, 2, null];
  const report = [2, 3, 4, 5, 6, null, 8];
  const a = M.imputeLinelist({ onset, report, maxDelay: 10 });
  assert.ok(finite(a));
  assert.equal(a.cutoff, 8);
  assert.equal(a.imputed.fwd, null);
  assert.equal(a.droppedImputed.fwd, null);
  const b = M.imputeLinelist({ onset, report, delay: [0.2, 0.3, 0.5] });
  assert.ok(finite(b) && b.imputed.fwd.every(Number.isFinite));
  assert.ok(Math.abs(sum(b.generativeDelay) - 1) < 1e-12);
});

test("imputed onsets before day 0 are counted, not lost silently", () => {
  // complete cases all have a delay of 3; missing cases reported on days
  // 0 to 2 would get onsets before day 0
  const onset = [0, 1, 2, 3, null, null, null, null];
  const report = [3, 4, 5, 6, 0, 1, 2, 6];
  const res = M.imputeLinelist({ onset, report, iterations: 0 });
  assert.deepEqual(res.empiricalDelay.slice(0, 4), [0, 0, 0, 1]);
  assert.equal(res.droppedImputed.backsampling, 3);
  assert.equal(res.droppedImputed.mean, 3);
  for (const k of ["backsampling", "mean"]) {
    const kept = sum(res.imputed[k]) - sum(res.complete);
    assert.equal(kept + res.droppedImputed[k], sum(res.missingByReport), k);
  }
  // with a given delay, the forward draw is counted the same way
  // half the mass of the case reported on day 0 falls on day -1
  const fwd = M.imputeLinelist({ onset, report, delay: [0.5, 0.5] });
  assert.equal(fwd.droppedImputed.fwd, 0.5);
  const fwd3 = M.imputeLinelist({ onset, report, delay: [0, 0, 0, 1] });
  assert.equal(fwd3.droppedImputed.fwd, 3);
  // `dropped` still counts input cases only
  assert.equal(res.dropped, 0);
});

test("no complete cases: nothing is NaN and every missing case is counted", () => {
  const res = M.imputeLinelist({
    onset: [null, null, null],
    report: [1, 2, 4]
  });
  assert.ok(finite(res));
  assert.deepEqual(res.empiricalDelay.slice(0, 3), [0, 0, 0]);
  assert.equal(res.droppedImputed.backsampling, 3);
  // EM estimates a delay from the missing cases alone, so it places them
  assert.ok(Math.abs(sum(res.imputed.gen) + res.droppedImputed.gen - 3) < 1e-9);
  assert.equal(sum(res.imputed.backsampling), 0);
});

test("an empty line list with a cut-off gives zeros", () => {
  const res = M.imputeLinelist({ onset: [], report: [], cutoff: 3 });
  assert.equal(res.days, 4);
  assert.deepEqual(res.imputed.gen, [0, 0, 0, 0]);
  assert.equal(res.dropped, 0);
});
