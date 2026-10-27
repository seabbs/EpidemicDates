// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Shared helpers for the Node tests. Not a test file itself.
import { createRequire } from "node:module";

/** The module under test, loaded as Node users load it. */
export const M = createRequire(import.meta.url)("../onset-imputation.js");

/**
 * Run the expected-value pipeline step by step, returning every stage.
 * @param {object} set Scenario parameters to merge with the defaults.
 */
export function pipeline(set) {
  const p = { ...M.DEFAULTS, ...set };
  const sc = M.scenario(p);
  const ex = M.expectedData(sc);
  const R = M.runMethods(sc, ex.comp, ex.miss, ex.empCounts);
  const W = Math.floor(Math.min(sc.N, sc.Td + 1) / 7);
  const tw = M.weekly(ex.truth, W, sc.B + sc.wk0);
  const mw = {};
  for (const m of M.METHODS) mw[m.key] = M.weekly(R[m.key], W, sc.B + sc.wk0);
  return { p, sc, ex, R, W, tw, mw };
}

/** Largest absolute weekly relative error over weeks with 10 or more cases. */
export const maxRelErr = (mw, tw) =>
  Math.max(
    ...Array.from(mw, (v, w) => (tw[w] >= 10 ? Math.abs(v / tw[w] - 1) : 0))
  );

// longer delays and more missing onsets than the baseline, so effects are
// large enough to test
export const LONG = { mu0: 8, mu1: 8, cv: 1, pi0: 0.4, pi1: 0.4 };
export const steady = r => ({ knots: M.KNOTS.map(() => r), ...LONG });
export const bvd = M.PRESETS[0].set;
export const preset = name => M.PRESETS.find(p => p.name === name).set;

/**
 * A line list with known onsets: expected counts per (onset, delay) cell,
 * rounded, with every 5th and 5th + 1 case missing its onset.
 */
export function linelist(set, cutoff) {
  const p = { ...M.DEFAULTS, ...set };
  const sc = M.scenario(p);
  const onset = [],
    report = [],
    truth = [];
  let i = 0;
  for (let s = sc.B; s < sc.S; s++) {
    for (let d = 0; d <= M.D; d++) {
      const t = s + d;
      if (t - sc.B > cutoff) break;
      const n = Math.round(sc.C[s] * sc.f[s][d]);
      for (let k = 0; k < n; k++, i++) {
        report.push(t - sc.B);
        onset.push(i % 5 < 2 ? null : s - sc.B);
        truth.push(s - sc.B);
      }
    }
  }
  return { onset, report, truth };
}

/** Sum daily counts into weeks starting on day 0. */
export const byWeek = (daily, W) =>
  Array.from({ length: W }, (_, w) =>
    daily.slice(7 * w, 7 * w + 7).reduce((a, b) => a + b, 0)
  );

export const sum = a => Array.from(a).reduce((x, y) => x + y, 0);
