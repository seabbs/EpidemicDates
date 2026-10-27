// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Page logic for index.html: controls, charts and the step-through.
"use strict";
// ---------- maths from epidemic-dates.js ----------
const {
  D,
  KNOTS,
  R_MIN,
  R_MAX,
  DEFAULTS,
  PRESETS,
  rToR,
  wrongSideOf1,
  clamp,
  normalise,
  pmfMean,
  scenario,
  expectedData,
  runMethods,
  weekly,
  growth,
  stochastic,
  metrics,
  completeWeeks,
  comparisonSeries,
  emImpute,
  correctTruncation
} = EpidemicDates;
// line styles keep overlapping methods visible
const STYLE = {
  hub: { col: "--series-1", width: 4 },
  mean: { col: "--series-2", width: 2 },
  fwd: { col: "--series-3", width: 2, dash: "6 4" },
  gen: { col: "--series-4", width: 2.5, dash: "1 4" }
};
const METHODS = EpidemicDates.METHODS.map(m => ({ ...m, ...STYLE[m.key] }));

// ---------- page state and controls spec ----------
const CONTROLS = [
  {
    group: "Growth rate of onsets",
    editor: {
      id: "rEditor",
      hint: "Drag or sweep across the curve to set r(t). Points every 15 days."
    },
    items: [
      {
        id: "rwSd",
        label: "Random wobble in r",
        min: 0,
        max: 0.01,
        step: 0.0005,
        fmt: v => v.toFixed(4)
      },
      {
        id: "seed",
        label: "Random seed",
        min: 1,
        max: 50,
        step: 1,
        fmt: v => v
      },
      {
        id: "O0",
        label: "Onsets per day at day 0",
        min: 0.2,
        max: 30,
        step: 0.1,
        fmt: v => v.toFixed(1)
      },
      {
        id: "days",
        label: "Days shown",
        min: 70,
        max: 238,
        step: 7,
        fmt: v => v
      }
    ]
  },
  {
    group: "Onset-to-report delay",
    items: [
      {
        id: "mu0",
        label: "Mean delay at day 0 (days)",
        min: 1,
        max: 20,
        step: 0.5,
        fmt: v => v.toFixed(1)
      },
      {
        id: "mu1",
        label: "Mean delay at end (days)",
        min: 1,
        max: 20,
        step: 0.5,
        fmt: v => v.toFixed(1)
      },
      {
        id: "cv",
        label: "Coefficient of variation",
        min: 0.3,
        max: 1.6,
        step: 0.05,
        fmt: v => v.toFixed(2)
      }
    ]
  },
  {
    group: "Missing onset dates",
    items: [
      {
        id: "pi0",
        label: "Fraction missing at start",
        min: 0,
        max: 0.95,
        step: 0.05,
        fmt: v => Math.round(v * 100) + "%"
      },
      {
        id: "pi1",
        label: "Fraction missing at end",
        min: 0,
        max: 0.95,
        step: 0.05,
        fmt: v => Math.round(v * 100) + "%"
      },
      {
        id: "beta",
        label: "Extra missingness with long delays",
        min: 0,
        max: 3,
        step: 0.1,
        fmt: v => v.toFixed(1),
        hint: "0 is missing at random."
      }
    ]
  },
  {
    group: "Case ascertainment",
    editor: {
      id: "aEditor",
      hint: "Relative to day 0, by onset day. 1 is no change."
    },
    items: []
  },
  {
    group: "Data cut-off",
    items: [
      {
        id: "retro",
        type: "check",
        label: "Retrospective (all cases eventually reported)"
      },
      {
        id: "nowcast",
        type: "check",
        label: "Correct for right truncation (nowcast with the true delay)"
      },
      {
        id: "ended",
        type: "check",
        label: "Outbreak over (no onsets after the days shown)"
      },
      {
        id: "cutoff",
        label: "Real-time cut-off (day)",
        min: 40,
        max: 238,
        step: 1,
        fmt: v => v,
        hint: "At the far right, the cut-off is the last day shown."
      }
    ]
  },
  {
    group: "Noise",
    items: [
      {
        id: "stoch",
        type: "check",
        label: "Add stochastic replicates (Poisson)"
      },
      {
        id: "reps",
        label: "Replicates",
        min: 10,
        max: 100,
        step: 10,
        fmt: v => v
      }
    ]
  }
];
let P = { ...DEFAULTS, ...PRESETS[0].set };
P.knots = P.knots.slice();
P.asc = P.asc.slice();
let activePreset = 0;

// ---------- chart drawing ----------
const css = name =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) {
    hi = lo + 1;
  }
  const span = hi - lo,
    step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    t.push(+v.toFixed(10));
  return t;
}
function logTicks(lo, hi) {
  const t = [];
  for (
    let e = Math.floor(Math.log10(lo));
    e <= Math.ceil(Math.log10(hi));
    e++
  ) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= lo && v <= hi) t.push(v);
    }
  }
  return t;
}
const fmtNum = v => {
  if (v === null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-GB");
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(3);
};
function lineChart(el, o) {
  const width = Math.max(el.clientWidth, 280);
  const height = o.height || 260;
  const m = { l: 52, r: 14, t: 10, b: 36 };
  const iw = width - m.l - m.r,
    ih = height - m.t - m.b;
  const xs = [];
  for (const s of o.series)
    for (const p of s.data) if (p.y !== null && isFinite(p.y)) xs.push(p.x);
  if (!xs.length) {
    el.innerHTML = "<p class='sub'>No data.</p>";
    return;
  }
  const x0 = o.xmin ?? Math.min(...xs),
    x1 = o.xmax ?? Math.max(...xs);
  // y range from the points in view, so zooming rescales the y-axis
  const ys = [];
  for (const s of o.series)
    for (const p of s.data)
      if (p.y !== null && isFinite(p.y) && p.x >= x0 && p.x <= x1) ys.push(p.y);
  for (const b of o.bands || [])
    for (const p of b.data) if (p.x >= x0 && p.x <= x1) ys.push(p.lo, p.hi);
  if (!ys.length) ys.push(0, 1);
  let y0, y1;
  if (o.log) {
    const pos = ys.filter(v => v > 0);
    y0 = Math.max(Math.min(...pos), 0.5);
    y1 = Math.max(...pos) * 1.1;
  } else {
    y0 = Math.min(...ys, o.includeZero ? 0 : Infinity, o.refY ?? Infinity);
    y1 = Math.max(...ys, o.includeZero ? 0 : -Infinity, o.refY ?? -Infinity);
    if (o.yclamp) {
      y0 = Math.max(y0, o.yclamp[0]);
      y1 = Math.min(y1, o.yclamp[1]);
    }
    const pad = (y1 - y0) * 0.06 || 1;
    y0 -= pad;
    y1 += pad;
    if (o.includeZero && Math.min(...ys) >= 0) y0 = 0;
  }
  const sx = x => m.l + ((x - x0) / (x1 - x0 || 1)) * iw;
  const sy = o.log
    ? y =>
        m.t +
        ih -
        ((Math.log(Math.max(y, y0)) - Math.log(y0)) /
          (Math.log(y1) - Math.log(y0))) *
          ih
    : y => m.t + ih - ((clamp(y, y0, y1) - y0) / (y1 - y0)) * ih;
  const yt = o.log ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  const xt = niceTicks(x0, x1, Math.max(3, Math.floor(iw / 80)));
  const clipId = "clip" + Math.random().toString(36).slice(2, 8);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${o.aria || ""}"><defs><clipPath id="${clipId}"><rect x="${m.l}" y="${m.t - 2}" width="${iw}" height="${ih + 4}"/></clipPath></defs>`;
  for (const sh of o.shades || []) {
    const a = Math.max(sh.from, x0),
      b = Math.min(sh.to, x1);
    if (b > a)
      svg +=
        `<rect x="${sx(a)}" y="${m.t}" width="${sx(b) - sx(a)}" height="${ih}" fill="${sh.fill}"/>` +
        (sh.label
          ? `<text class="axis-label" x="${sx(a) + 4}" y="${m.t + ih - 6}">${sh.label}</text>`
          : "");
  }
  if (o.shade)
    svg += `<rect x="${sx(Math.max(o.shade[0], x0))}" y="${m.t}" width="${Math.max(0, sx(Math.min(o.shade[1], x1)) - sx(Math.max(o.shade[0], x0)))}" height="${ih}" fill="var(--shade)"/>`;
  for (const v of yt)
    svg += `<line class="gridline" x1="${m.l}" x2="${m.l + iw}" y1="${sy(v)}" y2="${sy(v)}"/><g class="tick"><text x="${m.l - 6}" y="${sy(v) + 4}" text-anchor="end">${o.yfmt ? o.yfmt(v) : fmtNum(v)}</text></g>`;
  for (const v of xt)
    svg += `<g class="tick"><text x="${sx(v)}" y="${m.t + ih + 16}" text-anchor="middle">${v}</text></g>`;
  if (o.zero && y0 < 0 && y1 > 0)
    svg += `<line class="zero" x1="${m.l}" x2="${m.l + iw}" y1="${sy(0)}" y2="${sy(0)}"/>`;
  if (o.refY !== undefined && y0 < o.refY && y1 > o.refY)
    svg += `<line class="zero" x1="${m.l}" x2="${m.l + iw}" y1="${sy(o.refY)}" y2="${sy(o.refY)}"/>`;
  for (const v of o.vlines || []) {
    if (v.x < x0 || v.x > x1) continue;
    svg += `<line x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${m.t}" y2="${m.t + ih}" stroke="${v.color}" stroke-width="2" stroke-dasharray="5 3"/>`;
    svg += `<text class="axis-label" x="${sx(v.x) - 4}" y="${m.t + 10}" text-anchor="end">${v.label}</text>`;
  }
  if (o.vline !== undefined && o.vline >= x0 && o.vline <= x1)
    svg += `<line class="cross" x1="${sx(o.vline)}" x2="${sx(o.vline)}" y1="${m.t}" y2="${m.t + ih}"/><text class="axis-label" x="${sx(o.vline) - 4}" y="${m.t + 10}" text-anchor="end">cut-off</text>`;
  svg += `<text class="axis-label" x="${m.l + iw / 2}" y="${height - 4}" text-anchor="middle">${o.xlab}</text>`;
  svg += `<text class="axis-label" transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${o.ylab}</text>`;
  for (const b of o.bands || []) {
    if (!b.data.length) continue;
    const up = b.data.map(p => `${sx(p.x)},${sy(p.hi)}`).join(" ");
    const dn = b.data
      .slice()
      .reverse()
      .map(p => `${sx(p.x)},${sy(p.lo)}`)
      .join(" ");
    svg += `<polygon points="${up} ${dn}" fill="${b.color}" opacity="0.16" clip-path="url(#${clipId})"/>`;
  }
  for (const s of o.series) {
    let d = "",
      pen = false;
    for (const p of s.data) {
      if (p.y === null || !isFinite(p.y) || (o.log && p.y <= 0)) {
        pen = false;
        continue;
      }
      d += (pen ? "L" : "M") + sx(p.x).toFixed(1) + "," + sy(p.y).toFixed(1);
      pen = true;
    }
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} clip-path="url(#${clipId})"/>`;
  }
  for (const mk of o.markers || []) {
    if (mk.x < x0 || mk.x > x1) continue;
    svg += `<circle cx="${sx(mk.x)}" cy="${sy(mk.y)}" r="5" fill="none" stroke="${mk.color}" stroke-width="2"/>`;
  }
  svg += `<line class="cross xh" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" style="display:none"/>`;
  svg += `<rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="hit"/>`;
  svg += `</svg><div class="tooltip"></div>`;
  el.innerHTML = svg;
  const hit = el.querySelector(".hit"),
    xh = el.querySelector(".xh"),
    tip = el.querySelector(".tooltip");
  const allX = [...new Set(o.series.flatMap(s => s.data.map(p => p.x)))]
    .filter(x => x >= x0 && x <= x1)
    .sort((a, b) => a - b);
  const move = ev => {
    const rect = el.querySelector("svg").getBoundingClientRect();
    const px = ((ev.clientX - rect.left) * width) / rect.width;
    const xv = x0 + ((px - m.l) / iw) * (x1 - x0);
    let best = allX[0];
    for (const v of allX) if (Math.abs(v - xv) < Math.abs(best - xv)) best = v;
    xh.setAttribute("x1", sx(best));
    xh.setAttribute("x2", sx(best));
    xh.style.display = "";
    let html = `<div style="margin-bottom:4px;font-weight:600">${o.xtip ? o.xtip(best) : best}</div>`;
    for (const s of o.series) {
      const p = s.data.find(q => q.x === best);
      if (!p || p.y === null) continue;
      html += `<div class="row"><span class="sw" style="background:${s.color}"></span><span class="name">${s.name}</span><span class="val">${o.vfmt ? o.vfmt(p.y) : fmtNum(p.y)}</span></div>`;
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    const left = (sx(best) / width) * rect.width;
    const tw = tip.offsetWidth;
    tip.style.left =
      (left + 12 + tw > rect.width ? left - tw - 12 : left + 12) + "px";
    tip.style.top = "8px";
  };
  hit.addEventListener("mousemove", move);
  hit.addEventListener("touchstart", e => move(e.touches[0]), {
    passive: true
  });
  hit.addEventListener("touchmove", e => move(e.touches[0]), { passive: true });
  hit.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    xh.style.display = "none";
  });
  if (o.onZoom) {
    const svgEl = el.querySelector("svg");
    const toX = ev => {
      const rect = svgEl.getBoundingClientRect();
      const px = clamp(
        ((ev.clientX - rect.left) * width) / rect.width,
        m.l,
        m.l + iw
      );
      return { px, x: x0 + ((px - m.l) / iw) * (x1 - x0) };
    };
    let start = null,
      brush = null;
    hit.addEventListener("mousedown", ev => {
      ev.preventDefault();
      start = toX(ev);
      brush = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      brush.setAttribute("y", m.t);
      brush.setAttribute("height", ih);
      brush.setAttribute("fill", "var(--text-muted)");
      brush.setAttribute("opacity", "0.15");
      brush.setAttribute("pointer-events", "none");
      svgEl.appendChild(brush);
      const onMove = e => {
        const cur = toX(e);
        brush.setAttribute("x", Math.min(start.px, cur.px));
        brush.setAttribute("width", Math.abs(cur.px - start.px));
      };
      const onUp = e => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        const end = toX(e);
        brush.remove();
        if (Math.abs(end.px - start.px) > 6)
          o.onZoom([Math.min(start.x, end.x), Math.max(start.x, end.x)]);
        start = null;
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });
    hit.addEventListener("dblclick", () => o.onZoom(null));
  }
}
function legend(el, items) {
  el.innerHTML = items
    .map(
      i =>
        `<span><i class="${i.dash ? "dash" : ""}" style="border-color:${i.color}"></i>${i.name}</span>`
    )
    .join("");
}

// ---------- verdict ----------
function verdict(met) {
  const h = met.hub;
  const pct = v => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(0) + "%";
  const items = [];
  items.push(
    `Largest weekly error <b>${pct(h.maxRel)}</b>. Largest Rt error <b>${(h.maxR >= 0 ? "+" : "−") + Math.abs(h.maxR).toFixed(2)}</b>. Weeks with Rt on the wrong side of 1: <b>${h.wrongSide}</b>.`
  );
  if (h.peakHeight !== null)
    items.push(
      `Peak ${h.peakHeight < 0 ? "under" : "over"}stated by <b>${Math.abs(h.peakHeight * 100).toFixed(0)}%</b>${h.peakShift ? `, moved ${h.peakShift} days` : ""}.`
    );
  if (h.last !== null)
    items.push(
      `Last 3 weeks before the cut-off: <b>${pct(h.last)}</b> (mean shift ${pct(met.mean.last)}).`
    );
  document.getElementById("verdict").className = "verdict";
  document.getElementById("verdict").innerHTML =
    `<div><span class="tag">Hub method.</span> ${items.join(" ")}</div>`;
}

// ---------- main render ----------
let cache = { key: null };
// shared x range for the time-series charts, null when zoomed out
let ZOOM = null;
const setZoom = z => {
  ZOOM = z;
  document.getElementById("zoomReset").disabled = !z;
  schedule();
};
function compute() {
  const key = JSON.stringify(P);
  if (cache.key === key) return cache;
  const sc = scenario(P);
  const ex = expectedData(sc);
  const R = runMethods(sc, ex.comp, ex.miss, ex.empCounts);
  const W = Math.floor(Math.min(sc.N, sc.Td + 1) / 7);
  const st = P.stoch ? stochastic(sc, W, P.reps, P.nowcast && !P.retro) : null,
    stochMsg = "";
  cache = { key, sc, ex, R, W, st, stochMsg };
  return cache;
}
function render() {
  const t0 = performance.now();
  const { sc, ex, R, W, st, stochMsg } = compute();
  const xw = w => sc.wk0 + 7 * w + 3.5,
    x0w = sc.wk0,
    x1w = sc.wk0 + 7 * W;
  const cs = comparisonSeries(P, sc, ex, R);
  const tw = weekly(cs.truth, W, sc.B + sc.wk0),
    rw = weekly(ex.rep, W, sc.B + sc.wk0);
  const mw = {};
  for (const M of METHODS)
    mw[M.key] = weekly(cs.methods[M.key], W, sc.B + sc.wk0);
  const gT = growth(tw),
    gM = {};
  for (const M of METHODS) gM[M.key] = growth(mw[M.key]);
  const rIn = Array.from({ length: W }, (_, w) => {
    let s = 0;
    for (let d = 0; d < 7; d++) s += sc.r[sc.B + sc.wk0 + 7 * w + d] ?? 0;
    return s / 7;
  });
  const col = {};
  for (const M of METHODS) col[M.key] = css(M.col);
  const ink = css("--text-primary"),
    grey = css("--reports");
  const shade = null;
  // weeks less than 80% reported by the cut-off
  const ok = completeWeeks(sc, W, cs.nowcast ? 0.3 : 0.8);
  const firstBad = ok.indexOf(false);
  const shades =
    firstBad >= 0
      ? [
          {
            from: sc.wk0 + 7 * firstBad,
            to: x1w,
            fill: "var(--grid)",
            label: cs.nowcast ? "under 30% reported" : "under 80% reported"
          }
        ]
      : [];
  const vline = P.retro ? undefined : sc.Td;
  const xtip = x => `Week of days ${x - 3.5} to ${x + 2.5}`;

  // chart A
  const sA = [
    {
      name: "Reported (by report week)",
      color: grey,
      dash: "4 3",
      data: Array.from(rw, (y, w) => ({ x: xw(w), y }))
    },
    {
      name: "True onsets",
      color: ink,
      width: 2.5,
      data: Array.from(tw, (y, w) => ({ x: xw(w), y }))
    }
  ];
  for (const M of METHODS)
    sA.push({
      name: M.name,
      color: col[M.key],
      dash: M.dash,
      width: M.width,
      data: Array.from(mw[M.key], (y, w) => ({ x: xw(w), y }))
    });
  legend(
    document.getElementById("legendA"),
    sA.map(s => ({ name: s.name, color: s.color, dash: !!s.dash }))
  );
  lineChart(document.getElementById("chartA"), {
    series: sA,
    log: document.getElementById("logA").checked,
    xlab: "Day (onset or report week)",
    ylab: "Cases per week",
    shade,
    shades,
    vline,
    xtip,
    includeZero: true,
    xmin: ZOOM ? ZOOM[0] : x0w,
    xmax: ZOOM ? ZOOM[1] : x1w,
    onZoom: setZoom,
    aria: "Weekly onset counts, true and imputed"
  });

  // chart B
  const sB = METHODS.map(M => ({
    name: M.name,
    color: col[M.key],
    dash: M.dash,
    width: M.width,
    data: Array.from(mw[M.key], (y, w) => ({
      x: xw(w),
      y: tw[w] > 0 ? (y / tw[w] - 1) * 100 : null
    }))
  }));
  const bB = st
    ? METHODS.map(M => ({
        color: col[M.key],
        data: st.bands[M.key].map(p => ({
          x: p.x,
          lo: p.lo * 100,
          hi: p.hi * 100
        }))
      }))
    : [];
  legend(
    document.getElementById("legendB"),
    sB.map(s => ({ name: s.name, color: s.color, dash: !!s.dash }))
  );
  lineChart(document.getElementById("chartB"), {
    series: sB,
    bands: bB,
    zero: true,
    includeZero: true,
    xlab: "Onset week (day)",
    ylab: "Error (%)",
    shade,
    shades,
    vline,
    xtip,
    xmin: ZOOM ? ZOOM[0] : x0w,
    xmax: ZOOM ? ZOOM[1] : x1w,
    onZoom: setZoom,
    yclamp: [-100, 200],
    vfmt: v => (v >= 0 ? "+" : "") + v.toFixed(1) + "%",
    yfmt: v => v + "%",
    aria: "Relative error of imputed weekly counts"
  });

  // chart C
  const toRBand = b => b.map(p => ({ x: p.x, lo: rToR(p.lo), hi: rToR(p.hi) }));
  const sC = [
    {
      name: "Input",
      color: ink,
      dash: "2 3",
      width: 1.5,
      data: rIn.map((y, w) => ({ x: xw(w), y: rToR(y) }))
    },
    {
      name: "True onsets",
      color: ink,
      width: 2.5,
      data: gT.map((y, w) => ({ x: xw(w), y: rToR(y) }))
    }
  ];
  for (const M of METHODS)
    sC.push({
      name: M.name,
      color: col[M.key],
      dash: M.dash,
      width: M.width,
      data: gM[M.key].map((y, w) => ({ x: xw(w), y: rToR(y) }))
    });
  const bC = st
    ? [
        { color: ink, data: toRBand(st.truthBand) },
        { color: col.hub, data: toRBand(st.gbands.hub) }
      ]
    : [];
  legend(
    document.getElementById("legendC"),
    sC.map(s => ({ name: s.name, color: s.color, dash: !!s.dash }))
  );
  lineChart(document.getElementById("chartC"), {
    series: sC,
    bands: bC,
    refY: 1,
    xlab: "Onset week (day)",
    ylab: "Rt",
    shade,
    shades,
    vline,
    xtip,
    xmin: ZOOM ? ZOOM[0] : x0w,
    xmax: ZOOM ? ZOOM[1] : x1w,
    onZoom: setZoom,
    yclamp: [0, 4],
    vfmt: v => v.toFixed(3),
    yfmt: v => v.toFixed(1),
    aria: "Reproduction number estimated from weekly counts"
  });

  // chart E: growth rate error
  const sE = METHODS.map(M => ({
    name: M.name,
    color: col[M.key],
    dash: M.dash,
    width: M.width,
    data: gM[M.key].map((y, w) => ({
      x: xw(w),
      y: y === null || gT[w] === null ? null : rToR(y) - rToR(gT[w])
    }))
  }));
  // weeks where a method puts Rt on the other side of 1 from the truth
  const markE = [];
  for (const M of METHODS)
    gM[M.key].forEach((y, w) => {
      if (y === null || gT[w] === null || !ok[w] || tw[w] < 10) return;
      if (wrongSideOf1(y, gT[w]))
        markE.push({ x: xw(w), y: rToR(y) - rToR(gT[w]), color: col[M.key] });
    });
  const bE = st
    ? METHODS.map(M => ({ color: col[M.key], data: st.ebands[M.key] }))
    : [];
  legend(
    document.getElementById("legendE"),
    sE.map(s => ({ name: s.name, color: s.color, dash: !!s.dash }))
  );
  lineChart(document.getElementById("chartE"), {
    series: sE,
    bands: bE,
    zero: true,
    includeZero: true,
    markers: markE,
    xlab: "Onset week (day)",
    ylab: "Rt error",
    shade,
    shades,
    vline,
    xtip,
    xmin: ZOOM ? ZOOM[0] : x0w,
    xmax: ZOOM ? ZOOM[1] : x1w,
    onZoom: setZoom,
    yclamp: [-1.5, 1.5],
    vfmt: v => (v >= 0 ? "+" : "") + v.toFixed(3),
    yfmt: v => v.toFixed(2),
    aria: "Error in Rt estimated from imputed counts"
  });

  // metrics
  const met = {};
  for (const M of METHODS)
    met[M.key] = metrics(tw, mw[M.key], gT, gM[M.key], ok, W, P.retro);
  verdict(met);
  const pct = v =>
    v === null
      ? "–"
      : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(1) + "%";
  const days = v =>
    v === null ? "–" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1);
  let html = `<thead><tr><th>Method</th><th>Max weekly error</th><th>Mean |error|</th><th>Max Rt error</th><th>Weeks Rt on wrong side of 1</th><th>Peak shift (days)</th><th>Peak height</th>${P.retro ? "" : "<th>Last 3 weeks</th>"}</tr></thead><tbody>`;
  for (const M of METHODS) {
    const q = met[M.key];
    html += `<tr><td><span class="sw" style="display:inline-block;width:10px;height:3px;border-radius:2px;background:${col[M.key]};margin-right:6px;vertical-align:middle"></span>${M.name}</td>
      <td>${pct(q.maxRel)}</td>
      <td>${pct(q.meanAbs).replace("+", "")}</td>
      <td>${q.maxR >= 0 ? "+" : "−"}${Math.abs(q.maxR).toFixed(3)}</td>
      <td>${q.wrongSide}</td>
      <td>${days(q.peakShift)}</td>
      <td>${pct(q.peakHeight)}</td>
      ${P.retro ? "" : `<td>${pct(q.last)}</td>`}</tr>`;
  }
  html += `</tbody>`;
  document.getElementById("summary").innerHTML = html;

  drawSteps();

  const tot = tw.reduce((a, b) => a + b, 0);
  document.getElementById("status").textContent =
    `Expected cases in the onset weeks shown: ${Math.round(tot).toLocaleString("en-GB")}. Cut-off day ${sc.Td}${P.retro ? " (retrospective)" : ""}. Realised mean delay ${pmfMean(sc.fbar).toFixed(1)} days.${stochMsg} Rendered in ${(performance.now() - t0).toFixed(0)} ms.`;
}

// ---------- controls ----------
function buildControls() {
  const host = document.getElementById("controls");
  const pre = document.getElementById("presets");
  PRESETS.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = p.name;
    b.addEventListener("click", () => {
      activePreset = i;
      ZOOM = null;
      document.getElementById("zoomReset").disabled = true;
      P = { ...DEFAULTS, ...p.set };
      P.knots = P.knots.slice();
      P.asc = P.asc.slice();
      syncControls();
      schedule();
    });
    pre.appendChild(b);
  });
  for (const g of CONTROLS) {
    const fs = document.createElement("fieldset");
    fs.innerHTML = `<legend>${g.group}</legend>`;
    if (g.editor) {
      const ed = document.createElement("div");
      ed.innerHTML = `<p class="hint">${g.editor.hint}</p><div id="${g.editor.id}"></div>`;
      fs.appendChild(ed);
    }
    for (const c of g.items) {
      const div = document.createElement("div");
      div.className = "ctl";
      if (c.type === "check") {
        div.innerHTML = `<label class="check" style="justify-content:flex-start"><input type="checkbox" id="c_${c.id}"> ${c.label}</label>`;
      } else {
        div.innerHTML = `<label for="c_${c.id}">${c.label} <output id="o_${c.id}"></output></label><input type="range" id="c_${c.id}" min="${c.min}" max="${c.max}" step="${c.step}">${c.hint ? `<p class="hint">${c.hint}</p>` : ""}`;
      }
      fs.appendChild(div);
    }
    host.appendChild(fs);
  }
  for (const g of CONTROLS)
    for (const c of g.items) {
      const el = document.getElementById("c_" + c.id);
      el.addEventListener("input", () => {
        P[c.id] = c.type === "check" ? el.checked : +el.value;
        // a cut-off at or past the last day shown follows the window
        if (c.id === "cutoff" && P.cutoff >= P.days - 1) P.cutoff = null;
        if (c.id === "days" || c.id === "cutoff") syncControls();
        syncEnabled();
        if (c.type !== "check")
          document.getElementById("o_" + c.id).textContent = c.fmt(P[c.id]);
        activePreset = -1;
        markPreset();
        schedule();
      });
    }
  syncControls();
}
function markPreset() {
  [...document.querySelectorAll("#presets button")].forEach((b, i) =>
    b.classList.toggle("active", i === activePreset)
  );
  document.getElementById("presetNote").textContent =
    activePreset >= 0 ? PRESETS[activePreset].note : "Custom settings.";
}
function syncControls() {
  for (const g of CONTROLS)
    for (const c of g.items) {
      const el = document.getElementById("c_" + c.id);
      if (c.type === "check") el.checked = !!P[c.id];
      else if (c.id === "cutoff") {
        const v = P.cutoff ?? P.days - 1;
        el.value = v;
        document.getElementById("o_cutoff").textContent =
          P.cutoff === null || P.cutoff === undefined ? `${v} (last day)` : v;
      } else {
        el.value = P[c.id];
        document.getElementById("o_" + c.id).textContent = c.fmt(P[c.id]);
      }
    }
  syncEnabled();
  drawEditor();
  markPreset();
}
// ---------- curve editors (growth rate and ascertainment) ----------
const EDITORS = [
  {
    el: "rEditor",
    key: "knots",
    min: R_MIN,
    max: R_MAX,
    step: 0.002,
    ref: 0,
    grid: [-0.05, 0, 0.05, 0.1],
    tick: v => v.toFixed(2),
    name: "r"
  },
  {
    el: "aEditor",
    key: "asc",
    min: 0,
    max: 2,
    step: 0.05,
    ref: 1,
    grid: [0, 0.5, 1, 1.5, 2],
    tick: v => v.toFixed(1),
    name: "ascertainment"
  }
];
const ED = { w: 296, h: 150, l: 34, r: 8, t: 8, b: 20 };
function edScales(cfg) {
  const iw = ED.w - ED.l - ED.r,
    ih = ED.h - ED.t - ED.b;
  const xmax = KNOTS[KNOTS.length - 1];
  return {
    sx: t => ED.l + (t / xmax) * iw,
    sy: v => ED.t + ((cfg.max - v) / (cfg.max - cfg.min)) * ih,
    ix: px =>
      clamp(
        Math.round(((px - ED.l) / iw) * (KNOTS.length - 1)),
        0,
        KNOTS.length - 1
      ),
    iy: py =>
      clamp(
        cfg.max - ((py - ED.t) / ih) * (cfg.max - cfg.min),
        cfg.min,
        cfg.max
      )
  };
}
function drawEditor() {
  for (const cfg of EDITORS) {
    const el = document.getElementById(cfg.el);
    if (!el) continue;
    const { sx, sy } = edScales(cfg),
      vals = P[cfg.key];
    let g = `<svg viewBox="0 0 ${ED.w} ${ED.h}" style="width:100%;height:auto;touch-action:none;cursor:crosshair">`;
    for (const v of cfg.grid) {
      g += `<line x1="${ED.l}" x2="${ED.w - ED.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(${v === cfg.ref ? "--text-muted" : "--grid"})"/>`;
      g += `<text x="${ED.l - 4}" y="${sy(v) + 3}" text-anchor="end" font-size="9" fill="var(--text-muted)">${cfg.tick(v)}</text>`;
    }
    for (const t of [0, 60, 120, 180, 240])
      g += `<text x="${sx(t)}" y="${ED.h - 6}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${t}</text>`;
    if (P.days < 240)
      g += `<rect x="${sx(P.days)}" y="${ED.t}" width="${sx(240) - sx(P.days)}" height="${ED.h - ED.t - ED.b}" fill="var(--grid)" opacity="0.6"/>`;
    g += `<polyline fill="none" stroke="var(--text-primary)" stroke-width="2" points="${KNOTS.map((t, j) => `${sx(t)},${sy(vals[j])}`).join(" ")}"/>`;
    KNOTS.forEach((t, j) => {
      g += `<circle cx="${sx(t)}" cy="${sy(vals[j])}" r="4" fill="var(--surface-1)" stroke="var(--text-primary)" stroke-width="1.5"/>`;
    });
    g += `<text class="edVal" x="${ED.w - ED.r}" y="${ED.t + 9}" text-anchor="end" font-size="10" fill="var(--text-secondary)"></text></svg>`;
    el.innerHTML = g;
  }
}
function wireEditor() {
  for (const cfg of EDITORS) {
    const el = document.getElementById(cfg.el);
    let down = false;
    const setFrom = ev => {
      const svg = el.querySelector("svg"),
        rect = svg.getBoundingClientRect();
      const px = ((ev.clientX - rect.left) * ED.w) / rect.width,
        py = ((ev.clientY - rect.top) * ED.h) / rect.height;
      const { ix, iy } = edScales(cfg);
      const j = ix(px),
        v = Math.round(iy(py) / cfg.step) * cfg.step;
      P[cfg.key][j] = +v.toFixed(3);
      activePreset = -1;
      markPreset();
      drawEditor();
      el.querySelector(".edVal").textContent =
        `day ${KNOTS[j]}: ${cfg.name} = ${P[cfg.key][j].toFixed(3)}`;
      schedule();
    };
    el.addEventListener("pointerdown", ev => {
      down = true;
      el.setPointerCapture(ev.pointerId);
      setFrom(ev);
    });
    el.addEventListener("pointermove", ev => {
      if (down) setFrom(ev);
    });
    const up = () => {
      down = false;
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  }
}
function syncEnabled() {
  document.getElementById("c_cutoff").disabled = P.retro;
  document.getElementById("c_ended").disabled = !P.retro;
  document.getElementById("c_nowcast").disabled = P.retro;
}
let timer = null;
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(render, 40);
}
document.getElementById("tSel").addEventListener("input", schedule);
for (const b of document.querySelectorAll(".tab"))
  b.addEventListener("click", () => {
    for (const o of document.querySelectorAll(".tab")) {
      const on = o === b;
      o.classList.toggle("active", on);
      o.setAttribute("aria-selected", on);
      document.getElementById("tab-" + o.dataset.tab).hidden = !on;
    }
    schedule();
  });
document
  .getElementById("zoomReset")
  .addEventListener("click", () => setZoom(null));
document.getElementById("logA").addEventListener("change", schedule);
document.getElementById("themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur
    ? cur === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  schedule();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
let lastWidth = 0;
new ResizeObserver(entries => {
  const w = Math.round(entries[0].contentRect.width);
  if (w !== lastWidth) {
    lastWidth = w;
    schedule();
  }
}).observe(document.querySelector(".charts"));

// ---------- step-through of the methods on the current scenario ----------
const STEP = { method: "hub", step: 0, iter: 40, timer: null };
const STEP_TITLES = [
  "What is observed",
  "The delay the method uses",
  "Where the onset could be",
  "Impute the missing onsets",
  "Apply to every report day"
];
const NSTEP = STEP_TITLES.length;
let emCache = { key: null, snaps: [] };
// EM iterations from a start using only the complete cases, for animation
function emSnapshots(c) {
  if (emCache.key === c.key) return emCache.snaps;
  const snaps = [];
  emImpute(c.ex.comp, c.ex.miss, c.sc, c.ex.comp, 40, (it, out) =>
    snaps.push(out)
  );
  emCache = { key: c.key, snaps };
  return snaps;
}
// small seeded generator so the sampled cases stay put between redraws
function rng(seed) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x6d2b79f5) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}
const drawIdx = (w, u) => {
  let acc = 0;
  for (let i = 0; i < w.length; i++) {
    acc += w[i];
    if (u <= acc) return i;
  }
  return w.length - 1;
};
// a sample of individual cases from the scenario: cases reported in the
// three weeks up to day t, plus three missing-onset cases reported on day t
function sampleCases(sc, t, bn) {
  const ti = t + sc.B,
    u = rng(1000 + t),
    cells = [],
    wts = [];
  for (let r = ti - 20; r <= ti - 1; r++)
    for (let d = 0; d <= 30; d++) {
      const s = r - d;
      if (s < 0) continue;
      cells.push([s, d, r]);
      wts.push(sc.C[s] * sc.f[s][d]);
    }
  const wn = normalise(wts),
    cases = [];
  for (let k = 0; k < 13; k++) {
    const [s, d, r] = cells[drawIdx(wn, u())];
    cases.push({
      onset: s - sc.B,
      report: r - sc.B,
      missing: u() < sc.pim(r, d)
    });
  }
  cases.sort((x, y) => x.report - y.report || x.onset - y.onset);
  for (let k = 0; k < 3; k++)
    cases.push({
      onset: t - drawIdx(bn, u()),
      report: t,
      missing: true,
      focus: true,
      u: u()
    });
  return cases;
}
function stepText(key, step, v) {
  const nm = METHODS.find(m => m.key === key).name;
  if (step === 0)
    return `A sample of cases from this scenario. Each has an onset (dot) and a report day (square). Some have no onset date (?). On report day ${v.t}, about ${((100 * v.missT) / Math.max(v.repT, 1e-9)).toFixed(0)}% of cases have no onset date. The three at the bottom are the ones we follow.`;
  if (step === 1) {
    if (key === "hub")
      return `The hub method pools the delays of every case with both dates in the data so far, mean ${v.gMean.toFixed(1)} days. Ink is the true delay for missing cases reported on day ${v.t}, mean ${v.bMean.toFixed(1)} days. It depends on how onsets were changing before day ${v.t}. The pooled delay is the same whatever day a case was reported.`;
    if (key === "mean")
      return `The mean shift keeps only the mean of the pooled delays, ${v.gMean.toFixed(1)} days, rounded to ${v.m}. Ink is the true delay for missing cases reported on day ${v.t}, mean ${v.bMean.toFixed(1)} days.`;
    if (key === "fwd")
      return `The forward draw uses the delay from onset to report, mean ${v.fMean.toFixed(1)} days, as estimated with corrections for truncation and censoring. Ink is the true delay for missing cases reported on day ${v.t}, mean ${v.bMean.toFixed(1)} days. The forward delay ignores how many onsets there were on each earlier day.`;
    return `The generative method uses the forward delay, mean ${v.fMean.toFixed(1)} days, with its own estimate of the onset curve. Ink is the true delay for missing cases reported on day ${v.t}, mean ${v.bMean.toFixed(1)} days.`;
  }
  if (step === 2) {
    if (key === "mean")
      return `Every missing case reported on day ${v.t} goes to day ${v.t - v.m}. Ink is where such cases truly had their onsets.`;
    if (key === "gen")
      return `Each onset day s gets weight λ(s) × f(${v.t} − s): the estimated onsets that day times the chance of that delay. Ink is where such cases truly had their onsets.`;
    return `${nm} gives each onset day the probability of the matching delay. Ink is where such cases truly had their onsets.`;
  }
  if (step === 3) {
    if (key === "hub" || key === "fwd")
      return `Each missing case gets one random delay from the distribution, so one imputed onset (filled dot). The ring is its true onset, which real data would not show.`;
    if (key === "mean")
      return `Each missing case moves back exactly ${v.m} days (filled dot). The ring is its true onset, which real data would not show.`;
    return `Each missing case is shared across onset days in proportion to the weights (dots sized by share). The ring is its true onset, which real data would not show.`;
  }
  if (key === "gen")
    return `Doing this for every report day gives the imputed onset curve. The generative method then re-estimates the onset curve and repeats. Iteration ${v.iter} of 40: press Run to watch it converge from a start using the complete cases only.${v.nowcast ? " Curves are corrected for right truncation." : ""}`;
  return `Doing this for every report day gives the imputed onset curve, here against the truth.${v.nowcast ? " Both are corrected for right truncation." : ""}`;
}
function drawCases(el, o) {
  const { cases, t, key, step, color, weights, truthW, dist, m } = o;
  const W = 900,
    rowH = 13,
    top = 10,
    from = t - 32,
    to = t + 1,
    x0 = 70;
  const X = d => x0 + ((d - from) * (W - x0 - 20)) / (to - from);
  const axisY = top + cases.length * rowH + 6,
    lowTop = axisY + 30,
    lowBase = lowTop + 90,
    H = lowBase + 30;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Individual cases and imputation">`;
  for (let d = from; d <= to; d++) {
    if ((d - t) % 5 !== 0) continue;
    g += `<line x1="${X(d)}" x2="${X(d)}" y1="${top - 4}" y2="${axisY}" stroke="var(--grid)"/>`;
    g += `<text x="${X(d)}" y="${axisY + 13}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${d}</text>`;
  }
  g += `<text x="${x0 - 8}" y="${axisY + 13}" text-anchor="end" font-size="10" fill="var(--text-muted)">day</text>`;
  cases.forEach((cs, i) => {
    const y = top + i * rowH + 4,
      faded = step >= 2 && !cs.focus;
    const op = faded ? 0.3 : 1;
    if (!cs.missing) {
      const hl = step === 1 && (key === "hub" || key === "mean");
      g += `<line x1="${X(cs.onset)}" x2="${X(cs.report)}" y1="${y}" y2="${y}" stroke="${hl ? color : "var(--text-secondary)"}" stroke-width="${hl ? 3 : 2}" opacity="${op}"/>`;
      g += `<circle cx="${X(cs.onset)}" cy="${y}" r="4" fill="var(--text-primary)" opacity="${op}"/>`;
    } else {
      g += `<text x="${X(cs.report) - 9}" y="${y + 4}" text-anchor="end" font-size="11" font-weight="700" fill="${cs.focus ? color : "var(--text-muted)"}" opacity="${op}">?</text>`;
    }
    g += `<rect x="${X(cs.report) - 4}" y="${y - 4}" width="8" height="8" fill="${cs.missing && cs.focus ? color : "var(--text-secondary)"}" opacity="${op}"/>`;
    if (cs.focus && step >= 3) {
      g += `<circle cx="${X(cs.onset)}" cy="${y}" r="6" fill="none" stroke="var(--text-primary)" stroke-width="1.5"/>`;
      if (key === "gen") {
        weights.forEach((w, d) => {
          if (w > 0.02)
            g += `<circle cx="${X(t - d)}" cy="${y}" r="${1.5 + 9 * Math.sqrt(w)}" fill="${color}" opacity="0.55"/>`;
        });
      } else {
        const d = key === "mean" ? m : drawIdx(weights, cs.u);
        g += `<line x1="${X(t - d)}" x2="${X(t)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2" stroke-dasharray="4 3"/>`;
        g += `<circle cx="${X(t - d)}" cy="${y}" r="4.5" fill="${color}"/>`;
      }
    }
  });
  if (step >= 1) {
    const bars = step === 1 ? dist : weights,
      maxW = Math.max(...bars.slice(0, 33), ...truthW.slice(0, 33), 0.01);
    const bh = w => ((lowBase - lowTop) * w) / maxW;
    const label =
      step === 1 ? "delay (days back from day " + t + ")" : "P(onset day)";
    for (let d = 0; d <= 32; d++) {
      if (key === "mean" && step >= 1) continue;
      g += `<rect x="${X(t - d) - 6}" y="${lowBase - bh(bars[d] || 0)}" width="12" height="${bh(bars[d] || 0)}" fill="${color}" opacity="0.75"/>`;
    }
    if (key === "mean")
      g += `<line x1="${X(t - m)}" x2="${X(t - m)}" y1="${lowTop}" y2="${lowBase}" stroke="${color}" stroke-width="4"/>`;
    const pts = truthW
      .slice(0, 33)
      .map((w, d) => `${X(t - d)},${lowBase - bh(w)}`)
      .join(" ");
    g += `<polyline points="${pts}" fill="none" stroke="var(--text-primary)" stroke-width="2.5"/>`;
    g += `<text x="${x0 - 8}" y="${lowTop + 10}" text-anchor="end" font-size="10" fill="var(--text-muted)">${label}</text>`;
    g += `<line x1="${x0}" x2="${X(to)}" y1="${lowBase}" y2="${lowBase}" stroke="var(--text-muted)"/>`;
  }
  g += `</svg>`;
  el.innerHTML = g;
}
function drawSteps() {
  const el = document.getElementById("stepChart");
  if (!el || !cache.sc) return;
  const c = cache,
    { sc, ex, R } = c,
    key = STEP.method,
    step = STEP.step;
  const M = METHODS.find(x => x.key === key),
    color = css(M.col),
    ink = css("--text-primary"),
    grey = css("--reports");
  const tSel = document.getElementById("tSel");
  tSel.max = sc.Td;
  if (+tSel.value > sc.Td) tSel.value = sc.Td;
  const t = +tSel.value,
    ti = t + sc.B;
  document.getElementById("tSelOut").textContent = t;
  const back = new Float64Array(D + 1);
  for (let d = 0; d <= D; d++) {
    const s = ti - d;
    if (s >= 0 && s < sc.S) back[d] = sc.C[s] * sc.f[s][d] * sc.pim(ti, d);
  }
  const bn = normalise(back);
  const m = Math.round(pmfMean(R.g));
  const dist =
    key === "hub" || key === "mean" ? R.g : key === "fwd" ? sc.fbar : sc.f[ti];
  const w = new Float64Array(D + 1);
  for (let d = 0; d <= D; d++) {
    const s = ti - d;
    if (s < 0 || s >= sc.S) continue;
    if (key === "hub") w[d] = R.g[d];
    else if (key === "fwd") w[d] = sc.fbar[d];
    else if (key === "gen")
      w[d] = (R.gen[s] / Math.max(sc.cover[s], 1e-4)) * sc.f[s][d];
  }
  const weights = normalise(w);
  const v = {
    t,
    missT: ex.miss[ti] || 0,
    repT: ex.rep[ti] || 0,
    gMean: pmfMean(R.g),
    fMean: pmfMean(sc.fbar),
    bMean: pmfMean(bn),
    m,
    iter: STEP.iter,
    nowcast: !!P.nowcast && !P.retro
  };
  const legendEl = document.getElementById("stepLegend");
  if (step < 4) {
    const cases = sampleCases(sc, t, bn);
    legend(legendEl, [
      { name: "Truth for day " + t, color: ink },
      { name: M.name, color }
    ]);
    drawCases(el, {
      cases,
      t,
      key,
      step,
      color,
      weights: Array.from(weights),
      truthW: Array.from(bn),
      dist: Array.from(dist),
      bn,
      m
    });
  } else {
    const day = (arr, from, to) => {
      const out = [];
      for (let x = from; x <= to; x++)
        out.push({ x, y: arr[x + sc.B] ?? null });
      return out;
    };
    const corr = arr => (v.nowcast ? correctTruncation(arr, sc) : arr);
    let imputed = R[key];
    if (key === "gen") {
      const snaps = emSnapshots(c);
      imputed = snaps[Math.min(STEP.iter, snaps.length) - 1];
    }
    const truth = v.nowcast ? sc.C : ex.truth;
    const to = Math.min(sc.Td, sc.N - 1);
    const series = [
      { name: "Truth", color: ink, width: 3, data: day(truth, 0, to) },
      {
        name: key === "gen" ? `${M.name}, iteration ${STEP.iter}` : M.name,
        color,
        width: 2.5,
        data: day(corr(imputed), 0, to)
      }
    ];
    legend(
      legendEl,
      series.map(x => ({ name: x.name, color: x.color }))
    );
    lineChart(el, {
      series,
      xlab: "Onset day",
      ylab: "Onsets per day",
      xmin: 0,
      xmax: to,
      includeZero: true,
      height: 260,
      vlines: [{ x: t, color: grey, label: `day ${t}` }],
      aria: "Imputed onset curve against the truth",
      vfmt: x => x.toFixed(2)
    });
  }
  document.getElementById("stepTitle").textContent =
    `Step ${step + 1} of ${NSTEP}: ${STEP_TITLES[step]}`;
  document.getElementById("stepText").textContent = stepText(key, step, v);
  document.getElementById("stepPrev").disabled = step === 0;
  document.getElementById("stepNext").disabled = step === NSTEP - 1;
  for (const b of document.querySelectorAll("#stepMethods .mtab"))
    b.classList.toggle("active", b.dataset.m === key);
}
function stopRun() {
  clearTimeout(STEP.timer);
  STEP.timer = null;
  document.getElementById("stepRun").textContent = "▶ Run";
}
function runSteps() {
  if (STEP.timer) {
    stopRun();
    return;
  }
  document.getElementById("stepRun").textContent = "❚❚ Pause";
  if (STEP.step === NSTEP - 1 && (STEP.method !== "gen" || STEP.iter >= 40)) {
    STEP.step = 0;
    STEP.iter = 1;
  }
  if (STEP.step < NSTEP - 1) STEP.iter = 1;
  const tick = () => {
    if (STEP.step < NSTEP - 1) {
      STEP.step++;
      drawSteps();
      STEP.timer = setTimeout(tick, 1600);
      return;
    }
    if (STEP.method === "gen" && STEP.iter < 40) {
      STEP.iter++;
      drawSteps();
      STEP.timer = setTimeout(tick, STEP.iter < 10 ? 400 : 120);
      return;
    }
    stopRun();
  };
  drawSteps();
  STEP.timer = setTimeout(tick, 1600);
}
function wireSteps() {
  const host = document.getElementById("stepMethods");
  host.innerHTML = METHODS.map(
    m =>
      `<button type="button" class="mtab" data-m="${m.key}">${m.name}</button>`
  ).join("");
  for (const b of host.querySelectorAll(".mtab"))
    b.addEventListener("click", () => {
      stopRun();
      STEP.method = b.dataset.m;
      STEP.iter = 40;
      drawSteps();
    });
  document.getElementById("stepPrev").addEventListener("click", () => {
    stopRun();
    STEP.step = Math.max(0, STEP.step - 1);
    STEP.iter = 40;
    drawSteps();
  });
  document.getElementById("stepNext").addEventListener("click", () => {
    stopRun();
    STEP.step = Math.min(NSTEP - 1, STEP.step + 1);
    STEP.iter = 40;
    drawSteps();
  });
  document.getElementById("stepRun").addEventListener("click", runSteps);
}
buildControls();
wireEditor();
wireSteps();
render();
