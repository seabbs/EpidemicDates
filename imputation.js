/*
 * imputation.js
 *
 * Maths behind index.html: simulate an epidemic with reporting delays and
 * missing onset dates, impute the missing onsets four ways, and compare
 * with the true onsets of the same cases. Expected values unless
 * `stochastic()` is used.
 *
 * Works as a plain <script> (defines the global `Imputation`) and as a
 * CommonJS module in Node:
 *
 *   const Imputation = require("./imputation.js");
 *   const res = Imputation.run({
 *     knots: Imputation.knotsFrom([[0, 0.05], [60, 0.05], [120, -0.01]]),
 *     mu0: 8, mu1: 8, pi0: 0.4, pi1: 0.4
 *   });
 *   res.metrics.hub.maxRel; // largest weekly relative error, hub method
 *
 * See README.md for the parameters and outputs.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.Imputation = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // ---------- parameters ----------
  const D = 100; // maximum delay (days)
  // growth rate r(t) is set at a knot every 15 days and is linear in between
  const KNOTS = Array.from({ length: 17 }, (_, j) => 15 * j);
  const R_MIN = -0.08, R_MAX = 0.12;
  // value of a knot curve at day t, linear between knots, flat beyond
  const curveAt = (vals, t) => {
    if (t <= KNOTS[0]) return vals[0];
    if (t >= KNOTS[KNOTS.length - 1]) return vals[vals.length - 1];
    const j = Math.floor(t / 15);
    return vals[j] + (vals[j + 1] - vals[j]) * (t - KNOTS[j]) / 15;
  };
  // knot values from a few (day, value) anchor points
  const knotsFrom = pts => KNOTS.map(t => {
    if (t <= pts[0][0]) return pts[0][1];
    for (let j = 1; j < pts.length; j++) {
      if (t <= pts[j][0]) {
        const [a, ra] = pts[j - 1], [b, rb] = pts[j];
        return +(ra + (rb - ra) * (t - a) / (b - a)).toFixed(4);
      }
    }
    return pts[pts.length - 1][1];
  });
  const DEFAULTS = {
    days: 180, O0: 2, knots: knotsFrom([[0, 0.05], [60, 0.05], [110, -0.01], [140, -0.01], [175, 0.03]]), rwSd: 0, seed: 1,
    mu0: 8, mu1: 8, cv: 1.0, pi0: 0.4, pi1: 0.4, beta: 0,
    asc: KNOTS.map(() => 1), retro: true, cutoff: 110,
    stoch: false, reps: 30, ended: false
  };
  const BVD = { knots: knotsFrom([[0, 0.05], [60, 0.05], [110, -0.01], [140, -0.01], [175, 0.03]]), O0: 1 };
  const PRESETS = [
    { name: "BVD 2026: slowing, then rising", note: "Growth about 0.05/day (Rt about 1.8) slows to slight decline after the response, then picks up again at the end.",
      set: { ...BVD } },
    { name: "BVD 2026: slow, stall, rebound", note: "Growth slows, stalls near Rt = 1, picks up again, then is brought down. Four changes.",
      set: { knots: knotsFrom([[0, 0.05], [45, 0.05], [75, 0], [105, 0], [120, 0.03], [150, 0.03], [180, -0.03]]), O0: 1 } },
    { name: "BVD 2026: real time", note: "The slowing scenario, with data cut at day 120 after the turn.",
      set: { ...BVD, retro: false, cutoff: 120 } },
    { name: "BVD 2026: longer delays", note: "14-day mean delay and 70% of onsets missing.",
      set: { ...BVD, mu0: 14, mu1: 14, pi0: 0.7, pi1: 0.7 } },
    { name: "BVD 2026: longer delays, real time", note: "14-day mean delay, 70% missing, cut at day 130.",
      set: { ...BVD, mu0: 14, mu1: 14, pi0: 0.7, pi1: 0.7, retro: false, cutoff: 130 } },
    { name: "Two waves", note: "Growth, control, resurgence, control.",
      set: { knots: knotsFrom([[0, 0.06], [30, 0.06], [50, -0.03], [90, -0.03], [105, 0.05], [135, 0.05], [150, -0.04]]), O0: 3 } },
    { name: "Oscillating", note: "Growth swings up and down every few weeks, for example with stop-start response or new chains.",
      set: { knots: KNOTS.map(t => +(0.01 + 0.035 * Math.sin(2 * Math.PI * t / 60)).toFixed(4)), O0: 5 } },
    { name: "Noisy growth", note: "Slowing trend with random changes in growth from week to week.",
      set: { knots: knotsFrom([[0, 0.04], [180, -0.01]]), rwSd: 0.004, seed: 3, O0: 1 } },
    { name: "Sharp turnaround", note: "Fast growth reversed within a week by a strong intervention.",
      set: { knots: knotsFrom([[0, 0.06], [75, 0.06], [90, -0.04]]), O0: 0.5 } },
    { name: "Isiro 2012: slow, noisy", note: "Slow growth around 0.008/day with wobble. Few cases.",
      set: { knots: knotsFrom([[0, 0.01], [120, 0.005], [180, -0.01]]), rwSd: 0.003, seed: 7, O0: 1.5, stoch: true } },
    { name: "Delays shorten", note: "BVD shape, with the mean delay falling from 12 to 4 days.",
      set: { ...BVD, mu0: 12, mu1: 4 } },
    { name: "Late cases miss onsets", note: "BVD shape, but long-delay cases more often lack an onset date.",
      set: { ...BVD, beta: 1.5 } },
    { name: "Ascertainment jump", note: "BVD shape, with detection doubling over days 60 to 75.",
      set: { ...BVD, asc: knotsFrom([[0, 1], [60, 1], [75, 2]]) } },
    { name: "Ascertainment falls", note: "BVD shape, with detection halving over days 90 to 120, for example from insecurity.",
      set: { ...BVD, asc: knotsFrom([[0, 1], [90, 1], [120, 0.5]]) } },
    { name: "Steady growth", note: "Constant growth at 0.05/day.",
      set: { knots: KNOTS.map(() => 0.05) } },
    { name: "Steady, no growth", note: "Constant incidence, Rt = 1.",
      set: { knots: KNOTS.map(() => 0), O0: 20 } },
    { name: "Steady decline", note: "Constant decline at −0.03/day.",
      set: { knots: KNOTS.map(() => -0.03), O0: 40 } }
  ];

  const METHODS = [
    { key: "hub", name: "Hub: draw from pooled empirical delays" },
    { key: "mean", name: "Mean shift" },
    { key: "fwd", name: "Draw from forward delay" },
    { key: "gen", name: "Generative (EM back-projection)" }
  ];

  // ---------- maths helpers ----------
  function gammln(xx) {
    const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let x = xx, y = xx, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += cof[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }
  function gammp(a, x) {
    if (x <= 0) return 0;
    const lead = -x + a * Math.log(x) - gammln(a);
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 1000; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-13) break;
      }
      return sum * Math.exp(lead);
    }
    let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-13) break;
    }
    return 1 - Math.exp(lead) * h;
  }
  const pmfCache = new Map();
  // Double interval censored daily delay (primarycensored form): the
  // primary event is uniform within its day and the secondary event is
  // censored to its day. P(d) = G(d + 1) - 2 G(d) + G(d - 1), where
  // G(x) = integral of the Gamma CDF from 0 to x
  //      = x F(x; k, theta) - k theta F(x; k + 1, theta).
  // Truncated at D and renormalised. Its mean equals the Gamma mean.
  function delayPmf(mu, cv) {
    const key = mu.toFixed(3) + "|" + cv.toFixed(3);
    if (pmfCache.has(key)) return pmfCache.get(key);
    const k = 1 / (cv * cv), theta = mu * cv * cv;
    const G = x => x <= 0 ? 0 : x * gammp(k, x / theta) - k * theta * gammp(k + 1, x / theta);
    const pmf = new Float64Array(D + 1);
    let tot = 0;
    for (let d = 0; d <= D; d++) {
      pmf[d] = Math.max(G(d + 1) - 2 * G(d) + G(d - 1), 0);
      tot += pmf[d];
    }
    for (let d = 0; d <= D; d++) pmf[d] /= tot;
    pmfCache.set(key, pmf);
    return pmf;
  }
  const logistic = x => 1 / (1 + Math.exp(-x));
  // Rt from growth rate with a Gamma generation interval (Wallinga and
  // Lipsitch): R = (1 + r * scale)^shape. Ebola-like mean 15 days, SD 9.
  const GI = { mean: 15, sd: 9 };
  const GI_SHAPE = (GI.mean / GI.sd) ** 2, GI_SCALE = GI.sd ** 2 / GI.mean;
  // growth rates on opposite sides of 0 (Rt on opposite sides of 1), ignoring
  // differences within 0.0005/day of 0
  const wrongSideOf1 = (a, b) => (a > 0) !== (b > 0) && Math.max(Math.abs(a), Math.abs(b)) > 5e-4;
  const rToR = r => r === null ? null : Math.pow(Math.max(1 + r * GI_SCALE, 1e-6), GI_SHAPE);
  const logit = p => Math.log(p / (1 - p));
  const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
  const pmfMean = f => { let m = 0; for (let d = 0; d < f.length; d++) m += d * f[d]; return m; };

  // ---------- scenario ----------
  function scenario(p) {
    // Internal day index i = display day + B. The burn-in of B days at the
    // initial growth rate avoids an artificial start at day 0.
    const N = p.days, B = D;
    const Td = p.retro ? N - 1 + D : Math.min(p.cutoff, N - 1);
    const T = Td + B;
    const S = T + 1;
    const r = new Float64Array(S), C = new Float64Array(S);
    // piecewise-linear r(t) through the knots plus mean-reverting noise
    const kv = p.knots;
    let seed = p.seed * 2654435761 >>> 0;
    const unif = () => {
      seed = (seed + 0x6D2B79F5) >>> 0;
      let z = seed;
      z = Math.imul(z ^ (z >>> 15), z | 1);
      z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
      return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
    };
    let noise = 0;
    for (let i = 0; i < S; i++) {
      const t = i - B;
      let rt;
      if (t <= KNOTS[0]) rt = kv[0];
      else if (t >= KNOTS[KNOTS.length - 1]) rt = kv[kv.length - 1];
      else {
        const j = KNOTS.findIndex(k => k > t) - 1;
        rt = kv[j] + (kv[j + 1] - kv[j]) * (t - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j]);
      }
      if (t >= 0 && p.rwSd > 0) {
        const u1 = Math.max(unif(), 1e-12), u2 = unif();
        noise = 0.95 * noise + p.rwSd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      r[i] = rt + noise;
    }
    let logO = Math.log(p.O0) - kv[0] * B;
    for (let i = 0; i < S; i++) {
      if (i > 0) logO += r[i - 1];
      const asc = curveAt(p.asc, i - B);
      const over = p.retro && p.ended && i - B >= N;
      C[i] = over ? 0 : Math.exp(logO) * asc;
    }
    const f = [], F = [];
    for (let i = 0; i < S; i++) {
      const mu = p.mu0 + (p.mu1 - p.mu0) * clamp((i - B) / (N - 1), 0, 1);
      const pmf = delayPmf(mu, p.cv);
      f.push(pmf);
      const cdf = new Float64Array(D + 1);
      let acc = 0;
      for (let d = 0; d <= D; d++) { acc += pmf[d]; cdf[d] = acc; }
      F.push(cdf);
    }
    const muRef = p.mu0;
    // probability onset is missing, by internal report day and delay
    const pim = (i, d) => {
      const frac = clamp((i - B) / (N - 1), 0, 1);
      const base = clamp(p.pi0 + (p.pi1 - p.pi0) * frac, 0, 0.999);
      if (base <= 0) return 0;
      if (p.beta === 0) return base;
      return logistic(logit(base) + p.beta * (d - muRef) / muRef);
    };
    // true onset-weighted forward delay over the displayed onsets, standing in
    // for a delay estimated with a truncation adjustment
    const fbar = new Float64Array(D + 1);
    let cs = 0;
    for (let i = B; i < B + Math.min(N, Td + 1); i++) {
      cs += C[i];
      for (let d = 0; d <= D; d++) fbar[d] += C[i] * f[i][d];
    }
    for (let d = 0; d <= D; d++) fbar[d] /= cs;
    // expected fraction of each onset day's cases reported by the cut-off
    const cover = new Float64Array(S);
    for (let i = 0; i < S; i++) cover[i] = F[i][Math.min(T - i, D)];
    return { N, B, T, Td, S, r, C, f, F, pim, fbar, cover };
  }

  // ---------- expected-value data ----------
  function expectedData(sc) {
    const { T, S, C, f, pim } = sc;
    const truth = new Float64Array(S), comp = new Float64Array(S);
    const miss = new Float64Array(T + 1), rep = new Float64Array(T + 1);
    const empCounts = new Float64Array(D + 1);
    for (let s = 0; s < S; s++) {
      for (let d = 0; d <= D && s + d <= T; d++) {
        const x = C[s] * f[s][d], pm = pim(s + d, d);
        truth[s] += x; rep[s + d] += x;
        comp[s] += x * (1 - pm); miss[s + d] += x * pm;
        empCounts[d] += x * (1 - pm);
      }
    }
    return { truth, comp, miss, rep, empCounts };
  }

  // ---------- imputation methods (expected allocation) ----------
  function normalise(a) {
    let s = 0; for (const v of a) s += v;
    const out = new Float64Array(a.length);
    if (s > 0) for (let i = 0; i < a.length; i++) out[i] = a[i] / s;
    return out;
  }
  function convImpute(comp, miss, g, S) {
    const out = Float64Array.from(comp);
    for (let t = 0; t < miss.length; t++) {
      if (!miss[t]) continue;
      for (let d = 0; d <= D; d++) {
        const s = t - d;
        if (s >= 0 && s < S) out[s] += miss[t] * g[d];
      }
    }
    return out;
  }
  function shiftImpute(comp, miss, m, S) {
    const out = Float64Array.from(comp);
    for (let t = 0; t < miss.length; t++) {
      const s = t - m;
      if (s >= 0 && s < S) out[s] += miss[t];
    }
    return out;
  }
  function emImpute(comp, miss, sc, start) {
    const { S, T, f, F } = sc;
    const trunc = new Float64Array(S);
    for (let s = 0; s < S; s++) trunc[s] = Math.max(F[s][Math.min(T - s, D)], 1e-4);
    let lam = new Float64Array(S);
    for (let s = 0; s < S; s++) lam[s] = (start[s] + 1e-6) / trunc[s];
    let alloc = new Float64Array(S);
    for (let it = 0; it < 80; it++) {
      alloc = new Float64Array(S);
      for (let t = 0; t < miss.length; t++) {
        if (!miss[t]) continue;
        let den = 0;
        for (let d = 0; d <= D; d++) { const s = t - d; if (s >= 0 && s < S) den += lam[s] * f[s][d]; }
        if (den <= 0) continue;
        for (let d = 0; d <= D; d++) {
          const s = t - d;
          if (s >= 0 && s < S) alloc[s] += miss[t] * lam[s] * f[s][d] / den;
        }
      }
      for (let s = 0; s < S; s++) lam[s] = (comp[s] + alloc[s] + 1e-9) / trunc[s];
    }
    const out = new Float64Array(S);
    for (let s = 0; s < S; s++) out[s] = comp[s] + alloc[s];
    return out;
  }
  function runMethods(sc, comp, miss, empCounts) {
    const g = normalise(empCounts);
    const m = Math.round(pmfMean(g));
    const hub = convImpute(comp, miss, g, sc.S);
    return {
      hub,
      mean: shiftImpute(comp, miss, m, sc.S),
      fwd: convImpute(comp, miss, sc.fbar, sc.S),
      gen: emImpute(comp, miss, sc, hub),
      g, m
    };
  }

  // ---------- weekly summaries ----------
  function weekly(daily, W, off) {
    const out = new Float64Array(W);
    for (let w = 0; w < W; w++) for (let d = 0; d < 7; d++) out[w] += daily[off + 7 * w + d] || 0;
    return out;
  }
  // centred 3-week log-linear growth rate per day; off guards log(0) for
  // stochastic counts and is 0 for expected values
  // Weekly growth rate per day: centred 3-week log-linear slope, with a
  // one-sided 2-week slope in the first and last weeks so the series runs
  // to both ends. off guards log(0) for stochastic counts.
  function growth(wk, off = 0) {
    const n = wk.length, out = new Array(n).fill(null);
    const lg = w => wk[w] + off > 0 ? Math.log(wk[w] + off) : null;
    for (let w = 0; w < n; w++) {
      const a = Math.max(w - 1, 0), b = Math.min(w + 1, n - 1);
      if (b === a) continue;
      const la = lg(a), lb = lg(b);
      if (la === null || lb === null) continue;
      out[w] = (lb - la) / (7 * (b - a));
    }
    return out;
  }

  // ---------- stochastic replicates ----------
  function rpois(lam) {
    if (lam <= 0) return 0;
    if (lam > 40) return Math.max(0, Math.round(lam + Math.sqrt(lam) * randn()));
    const L = Math.exp(-lam); let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
  }
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sampleIdx(cdf) {
    const u = Math.random() * cdf[cdf.length - 1];
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function cumsum(a) { const c = new Float64Array(a.length); let s = 0; for (let i = 0; i < a.length; i++) { s += a[i]; c[i] = s; } return c; }
  function rbinom(n, p) {
    if (n <= 0 || p <= 0) return 0;
    if (p >= 1) return n;
    if (n < 40) { let k = 0; for (let i = 0; i < n; i++) if (Math.random() < p) k++; return k; }
    const m = n * p, v = n * p * (1 - p);
    if (m < 10) return Math.min(n, rpois(m));
    return clamp(Math.round(m + Math.sqrt(v) * randn()), 0, n);
  }
  // multinomial allocation of n cases over the probabilities in prob
  function rmultinom(n, prob) {
    const out = new Float64Array(prob.length);
    let left = n, rest = 1;
    for (let k = 0; k < prob.length && left > 0; k++) {
      const x = k === prob.length - 1 ? left : rbinom(left, rest > 0 ? clamp(prob[k] / rest, 0, 1) : 0);
      out[k] = x; left -= x; rest -= prob[k];
    }
    return out;
  }
  function stochastic(sc, W, reps) {
    const { S, T, C, f, pim } = sc;
    const relErr = {}, gr = {}, gerr = {};
    for (const M of METHODS) { relErr[M.key] = []; gr[M.key] = []; gerr[M.key] = []; }
    const grTruth = [];
    for (let rep = 0; rep < reps; rep++) {
      // Poisson thinning of each (onset, delay) cell keeps the cost at O(S * D)
      const truth = new Float64Array(S), comp = new Float64Array(S);
      const miss = new Float64Array(T + 1), empCounts = new Float64Array(D + 1);
      for (let s = 0; s < S; s++) {
        for (let d = 0; d <= D && s + d <= T; d++) {
          const x = C[s] * f[s][d];
          if (x <= 0) continue;
          const pm = pim(s + d, d);
          const nc = rpois(x * (1 - pm)), nm = rpois(x * pm);
          truth[s] += nc + nm; comp[s] += nc; miss[s + d] += nm; empCounts[d] += nc;
        }
      }
      const nEmp = empCounts.reduce((a, b) => a + b, 0);
      const g = normalise(empCounts);
      const res = {};
      // hub and forward: each missing case gets an independent delay draw
      res.hub = Float64Array.from(comp);
      res.fwd = Float64Array.from(comp);
      for (let t = 0; t <= T; t++) {
        if (!miss[t]) continue;
        const ah = nEmp ? rmultinom(miss[t], g) : new Float64Array(D + 1);
        const af = rmultinom(miss[t], sc.fbar);
        for (let d = 0; d <= D; d++) {
          const s = t - d;
          if (s < 0 || s >= S) continue;
          res.hub[s] += ah[d]; res.fwd[s] += af[d];
        }
      }
      res.mean = shiftImpute(comp, miss, Math.round(pmfMean(g)), S);
      res.gen = emImpute(comp, miss, sc, res.hub);
      const tw = weekly(truth, W, sc.B);
      const gtw = growth(tw, 0.5);
      grTruth.push(gtw);
      for (const M of METHODS) {
        const mw = weekly(res[M.key], W, sc.B);
        relErr[M.key].push(Array.from(mw, (v, w) => tw[w] >= 5 ? v / tw[w] - 1 : null));
        const gmw = growth(mw, 0.5);
        gr[M.key].push(gmw);
        gerr[M.key].push(gmw.map((v, w) => v === null || gtw[w] === null ? null : rToR(v) - rToR(gtw[w])));
      }
    }
    const q = (arrs, w, p) => {
      const v = arrs.map(a => a[w]).filter(x => x !== null && isFinite(x)).sort((a, b) => a - b);
      if (v.length < Math.max(3, reps / 3)) return null;
      const i = (v.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
      return v[lo] + (v[hi] - v[lo]) * (i - lo);
    };
    const bands = {}, gbands = {}, ebands = {};
    for (const M of METHODS) {
      bands[M.key] = []; gbands[M.key] = []; ebands[M.key] = [];
      for (let w = 0; w < W; w++) {
        const lo = q(relErr[M.key], w, 0.05), hi = q(relErr[M.key], w, 0.95);
        if (lo !== null) bands[M.key].push({ x: 7 * w + 3.5, lo, hi });
        const glo = q(gr[M.key], w, 0.05), ghi = q(gr[M.key], w, 0.95);
        if (glo !== null) gbands[M.key].push({ x: 7 * w + 3.5, lo: glo, hi: ghi });
        const elo = q(gerr[M.key], w, 0.05), ehi = q(gerr[M.key], w, 0.95);
        if (elo !== null) ebands[M.key].push({ x: 7 * w + 3.5, lo: elo, hi: ehi });
      }
    }
    const tb = [];
    for (let w = 0; w < W; w++) {
      const lo = q(grTruth, w, 0.05), hi = q(grTruth, w, 0.95);
      if (lo !== null) tb.push({ x: 7 * w + 3.5, lo, hi });
    }
    return { bands, gbands, ebands, truthBand: tb };
  }

  // ---------- metrics ----------
  // ok[w] marks onset weeks that are at least 80% reported by the cut-off.
  // Growth and peak metrics use only these, so truncation of the target
  // itself is not mistaken for a change in the epidemic.
  function metrics(tw, mw, gT, gM, ok, W, retro) {
    let maxRel = 0, maxRelW = -1, maxG = 0, maxR = 0, wrongSide = 0, sumAbs = 0, nAbs = 0;
    for (let w = 0; w < W; w++) {
      if (tw[w] < 10) continue;
      const rel = mw[w] / tw[w] - 1;
      sumAbs += Math.abs(rel); nAbs++;
      if (Math.abs(rel) > Math.abs(maxRel)) { maxRel = rel; maxRelW = w; }
    }
    for (let w = 0; w < W; w++) {
      if (!ok[w] || gT[w] === null || gM[w] === null || tw[w] < 10) continue;
      if (Math.abs(gM[w] - gT[w]) > Math.abs(maxG)) maxG = gM[w] - gT[w];
      const dR = rToR(gM[w]) - rToR(gT[w]);
      if (Math.abs(dR) > Math.abs(maxR)) maxR = dR;
      if (wrongSideOf1(gM[w], gT[w])) wrongSide++;
    }
    const firstNeg = g => {
      let lastPos = null;
      for (let w = 0; w < g.length; w++) {
        if (!ok[w] || g[w] === null) continue;
        if (g[w] > 0) lastPos = w;
        else if (lastPos !== null && g[w] < 0) {
          // linear interpolation of the zero crossing, in days
          const frac = g[lastPos] / (g[lastPos] - g[w]);
          return 7 * (lastPos + frac * (w - lastPos)) + 3.5;
        }
      }
      return null;
    };
    const zT = firstNeg(gT), zM = firstNeg(gM);
    const lag = zT !== null && zM !== null ? zM - zT : null;
    // peak metrics only when the true curve has an interior peak among
    // complete weeks, with a clear fall after it
    const nOk = ok.reduce((a, v, i) => (v ? i + 1 : a), 0);
    const argmax = a => { let b = 0; for (let i = 1; i < nOk; i++) if (a[i] > a[b]) b = i; return b; };
    const pT = argmax(tw);
    const hasPeak = nOk > 2 && pT < nOk - 2 && tw[nOk - 1] < 0.8 * tw[pT];
    const peakShift = hasPeak ? 7 * (argmax(mw) - pT) : null;
    const peakHeight = hasPeak ? mw[argmax(mw)] / tw[pT] - 1 : null;
    let last = null;
    if (!retro) {
      let s = 0, n = 0;
      for (let w = W - 3; w < W; w++) if (w >= 0 && tw[w] > 0) { s += mw[w] / tw[w] - 1; n++; }
      last = n ? s / n : null;
    }
    return { maxRel, maxRelW, maxG, maxR, wrongSide, meanAbs: nAbs ? sumAbs / nAbs : 0, lag, zT, zM, peakShift, peakHeight, last };
  }

  // ---------- convenience wrappers ----------
  // Onset weeks at least 80% reported by the cut-off (in expectation).
  function completeWeeks(sc, W) {
    return Array.from({ length: W }, (_, w) => {
      let num = 0, den = 0;
      for (let d = 0; d < 7; d++) {
        const i = sc.B + 7 * w + d;
        num += sc.C[i] * sc.cover[i]; den += sc.C[i];
      }
      return den > 0 ? num / den >= 0.8 : true;
    });
  }

  // Run one scenario end to end in expectation.
  // params: any subset of DEFAULTS. Returns weekly series for the truth,
  // reports and each method, growth rates, Rt and summary metrics.
  function run(params = {}) {
    const p = { ...DEFAULTS, ...params };
    const sc = scenario(p);
    const ex = expectedData(sc);
    const R = runMethods(sc, ex.comp, ex.miss, ex.empCounts);
    const W = Math.floor(Math.min(sc.N, sc.Td + 1) / 7);
    const ok = completeWeeks(sc, W);
    const truth = weekly(ex.truth, W, sc.B);
    const gT = growth(truth);
    const out = {
      params: p,
      weekStart: Array.from({ length: W }, (_, w) => 7 * w),
      complete: ok,
      truth: Array.from(truth),
      reported: Array.from(weekly(ex.rep, W, sc.B)),
      growth: { truth: gT },
      rt: { truth: gT.map(rToR) },
      imputed: {},
      metrics: {},
      pooledDelay: Array.from(R.g),
      forwardDelay: Array.from(sc.fbar)
    };
    for (const m of METHODS) {
      const mw = weekly(R[m.key], W, sc.B);
      const gM = growth(mw);
      out.imputed[m.key] = Array.from(mw);
      out.growth[m.key] = gM;
      out.rt[m.key] = gM.map(rToR);
      out.metrics[m.key] = metrics(truth, mw, gT, gM, ok, W, p.retro);
    }
    return out;
  }

  // ---------- real line lists ----------
  // Impute missing onsets in a real line list.
  // onset: onset day per case, null (or undefined) where missing.
  // report: report day per case. Days are integers, e.g. days since the
  //   first report. Cases reported after `cutoff` are dropped.
  // delay: optional forward delay pmf (array indexed by delay in days).
  //   If given, `fwd` uses it and `gen` holds it fixed. If not, `gen`
  //   estimates a single delay jointly with the onset curve, allowing for
  //   right truncation at the cut-off.
  // Returns daily onset counts (index = day) for complete cases and for
  // each method, plus the delays used. Counts are expected allocations.
  function imputeLinelist({ onset, report, cutoff = null, delay = null,
    maxDelay = D, iterations = 300 }) {
    if (onset.length !== report.length) throw new Error("onset and report must have the same length");
    const T = cutoff ?? Math.max(...report);
    const S = T + 1;
    const comp = new Float64Array(S), miss = new Float64Array(S);
    const emp = new Float64Array(maxDelay + 1);
    const Nc = Array.from({ length: S }, () => new Float64Array(maxDelay + 1));
    let dropped = 0;
    for (let i = 0; i < report.length; i++) {
      const t = report[i], o = onset[i];
      if (!Number.isInteger(t) || t < 0 || t > T) { dropped++; continue; }
      if (o === null || o === undefined) { miss[t]++; continue; }
      const d = t - o;
      if (!Number.isInteger(o) || o < 0 || d < 0 || d > maxDelay) { dropped++; continue; }
      comp[o]++; emp[d]++; Nc[o][d]++;
    }
    const g = normalise(emp);
    const hub = convImpute(comp, miss, g, S);
    const mean = shiftImpute(comp, miss, Math.round(pmfMean(g)), S);
    const fwd = delay ? convImpute(comp, miss, normalise(Float64Array.from(delay)), S) : null;
    // EM back-projection, optionally estimating a static delay
    let f = delay ? normalise(Float64Array.from({ length: maxDelay + 1 }, (_, d) => delay[d] ?? 0)) : Float64Array.from(g);
    let lam = Float64Array.from(hub, v => v + 1e-6);
    let alloc = new Float64Array(S);
    for (let it = 0; it < iterations; it++) {
      const Fc = cumsum(f);
      const A = Array.from({ length: S }, () => new Float64Array(maxDelay + 1));
      alloc = new Float64Array(S);
      for (let t = 0; t < S; t++) {
        if (!miss[t]) continue;
        let den = 0;
        for (let d = 0; d <= maxDelay && d <= t; d++) den += lam[t - d] * f[d];
        if (den <= 0) continue;
        for (let d = 0; d <= maxDelay && d <= t; d++) {
          const a = miss[t] * lam[t - d] * f[d] / den;
          A[t - d][d] += a; alloc[t - d] += a;
        }
      }
      for (let s = 0; s < S; s++) {
        let n = comp[s] + alloc[s];
        lam[s] = (n + 1e-9) / Math.max(Fc[Math.min(T - s, maxDelay)], 1e-4);
      }
      if (!delay) {
        // right-truncated delay update: cases with delay d can only be seen
        // for onsets up to T - d
        const cl = cumsum(lam);
        const nf = new Float64Array(maxDelay + 1);
        for (let d = 0; d <= maxDelay; d++) {
          if (T - d < 0) continue;
          let nd = 0;
          for (let s = 0; s + d <= T; s++) nd += Nc[s][d] + A[s][d];
          nf[d] = Math.max(nd / Math.max(cl[T - d], 1e-12), 1e-12);
        }
        f = normalise(nf);
      }
    }
    const gen = Float64Array.from(comp, (v, s) => v + alloc[s]);
    return {
      days: S, cutoff: T, dropped,
      complete: Array.from(comp), missingByReport: Array.from(miss),
      empiricalDelay: Array.from(g), generativeDelay: Array.from(f),
      imputed: {
        hub: Array.from(hub), mean: Array.from(mean),
        fwd: fwd ? Array.from(fwd) : null, gen: Array.from(gen)
      }
    };
  }

  return {
    D, KNOTS, R_MIN, R_MAX, GI, DEFAULTS, PRESETS, METHODS,
    knotsFrom, curveAt, delayPmf, pmfMean, rToR, wrongSideOf1, clamp, normalise,
    scenario, expectedData, runMethods, convImpute, shiftImpute, emImpute,
    weekly, growth, stochastic, metrics, completeWeeks, run, imputeLinelist
  };
});
