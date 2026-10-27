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
  knotsFrom,
  scenario,
  expectedData,
  runMethods,
  weekly,
  growth,
  stochastic,
  metrics,
  completeWeeks,
  comparisonSeries,
  maskWeeks,
  emImpute,
  correctTruncation
} = EpidemicDates;
// line styles keep overlapping methods visible
const STYLE = {
  backsampling: { col: "--series-1", width: 4 },
  mean: { col: "--series-2", width: 2 },
  fwd: { col: "--series-3", width: 2, dash: "6 4" },
  gen: { col: "--series-4", width: 2.5, dash: "1 4" }
};
const METHODS = EpidemicDates.METHODS.map(m => ({ ...m, ...STYLE[m.key] }));
// the last day the editors can set
const KNOT_END = KNOTS[KNOTS.length - 1];

// ---------- calendar dates ----------
// day 0 is the start date; internal maths stays in days
const START_DEFAULT = "2026-05-03";
let START = START_DEFAULT;
const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec"
];
const DAY_MS = 86400000;
function parseStart(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(s));
  if (!m) return null;
  const t = Date.UTC(+m[1], +m[2] - 1, +m[3]);
  return Number.isFinite(t) ? t : null;
}
const startMs = () => parseStart(START) ?? parseStart(START_DEFAULT);
const dateOf = day => new Date(startMs() + Math.round(day) * DAY_MS);
// "24 Oct 2026"
const fmtDate = day => {
  const d = dateOf(day);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};
// "24 Oct"
const fmtShort = day => {
  const d = dateOf(day);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
};
// tooltip title for a week centred on x
const fmtWeek = x => `Week of ${fmtShort(x - 3.5)} to ${fmtDate(x + 2.5)}`;
// x-axis ticks at month starts, thinned to fit; weekly for short ranges
function dateTicks(x0, x1, n) {
  const s0 = startMs();
  const a = dateOf(Math.floor(x0));
  let y = a.getUTCFullYear(),
    mo = a.getUTCMonth();
  const months = [];
  for (;;) {
    const x = (Date.UTC(y, mo, 1) - s0) / DAY_MS;
    if (x > x1) break;
    if (x >= x0) months.push({ x, y, mo });
    if (++mo === 12) {
      mo = 0;
      y++;
    }
  }
  if (months.length >= 2) {
    const k = Math.ceil(months.length / Math.max(n, 1));
    const kept = months.filter((_, i) => i % k === 0);
    return kept.map((t, i) => ({
      x: t.x,
      label: MONTHS[t.mo] + (i === 0 || t.mo === 0 ? " " + t.y : "")
    }));
  }
  const span = x1 - x0;
  const step = Math.max(7, 7 * Math.ceil(span / 7 / Math.max(n, 1)));
  const out = [];
  for (let x = Math.ceil(x0); x <= x1; x += step)
    out.push({ x, label: fmtShort(x) });
  return out;
}

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
        max: 357,
        step: 7,
        fmt: v => `${v} (to ${fmtShort(v - 1)})`
      }
    ]
  },
  {
    group: "Onset-to-report delay",
    editor: {
      id: "muEditor",
      hint: "Mean delay in days, by onset day."
    },
    items: [
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
    editor: {
      id: "piEditor",
      hint: "Fraction of cases with no onset date, by report day."
    },
    items: [
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
    group: "Dates and data cut-off",
    items: [
      { id: "start", type: "date", label: "Start date (day 0)" },
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
// a fresh copy of a preset, so editing never touches the module's constants
function fromPreset(set) {
  const p = { ...DEFAULTS, ...set };
  for (const k of ["knots", "asc", "muKnots", "piKnots"])
    if (p[k]) p[k] = p[k].slice();
  return p;
}
let P = fromPreset(PRESETS[0].set);
let activePreset = 0;
// series hidden from every chart, by legend name
const HIDDEN = new Set(["Input"]);
const shown = s => !HIDDEN.has(s.name);

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
const finite = v => v !== null && v !== undefined && isFinite(v);
// value at quantile p of a sorted array
const quantile = (a, p) => {
  if (!a.length) return 0;
  const i = (a.length - 1) * p,
    lo = Math.floor(i),
    hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
};
// y range for an error chart: at least ±min, widening with the data, but
// capped at 1.5 times the 95th centile of absolute values so one extreme
// point cannot flatten the rest
// error-chart ranges held fixed while "Lock the error axes" is ticked
const LOCKED = {};
function lockedRange(id) {
  const box = document.getElementById("lockAxes");
  if (!box || !box.checked) return null;
  if (!LOCKED[id]) {
    try {
      LOCKED[id] = JSON.parse(document.getElementById(id).dataset.range);
    } catch {
      return null;
    }
  }
  return LOCKED[id];
}
function robustRange(ys, min) {
  const abs = ys.map(Math.abs).sort((a, b) => a - b);
  const cap = Math.max(min, 1.5 * quantile(abs, 0.95));
  const lo = Math.min(...ys, 0),
    hi = Math.max(...ys, 0);
  return [Math.min(Math.max(lo, -cap), -min), Math.max(Math.min(hi, cap), min)];
}
function lineChart(el, o) {
  const width = Math.max(el.clientWidth, 280);
  const height = o.height || 260;
  const m = { l: 52, r: 14, t: 10, b: 36 };
  const iw = width - m.l - m.r,
    ih = height - m.t - m.b;
  const xs = [];
  for (const s of o.series)
    for (const p of s.data) if (finite(p.y)) xs.push(p.x);
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
      if (finite(p.y) && p.x >= x0 && p.x <= x1) ys.push(p.y);
  for (const b of o.bands || [])
    for (const p of b.data) if (p.x >= x0 && p.x <= x1) ys.push(p.lo, p.hi);
  if (!ys.length) ys.push(0, 1);
  let y0, y1;
  // values beyond the robust range are drawn at the edge with a label
  let edge = null;
  if (o.log) {
    const pos = ys.filter(v => v > 0);
    y0 = Math.max(Math.min(...pos), 0.5);
    y1 = Math.max(...pos) * 1.1;
  } else if (o.fixedRange) {
    [y0, y1] = o.fixedRange;
    edge = [y0, y1];
    const pad = (y1 - y0) * 0.06;
    y0 -= pad;
    y1 += pad;
  } else if (o.robust !== undefined) {
    [y0, y1] = robustRange(ys, o.robust);
    if (el.dataset) el.dataset.range = JSON.stringify([y0, y1]);
    edge = [y0, y1];
    const pad = (y1 - y0) * 0.06;
    y0 -= pad;
    y1 += pad;
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
  const nx = Math.max(3, Math.floor(iw / 80));
  const xt = o.dates
    ? dateTicks(x0, x1, nx)
    : niceTicks(x0, x1, nx).map(x => ({ x, label: x }));
  const xtip = o.xtip || (o.dates ? fmtDate : x => x);
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
  for (const v of yt)
    svg += `<line class="gridline" x1="${m.l}" x2="${m.l + iw}" y1="${sy(v)}" y2="${sy(v)}"/><g class="tick"><text x="${m.l - 6}" y="${sy(v) + 4}" text-anchor="end">${o.yfmt ? o.yfmt(v) : fmtNum(v)}</text></g>`;
  for (const v of xt)
    svg += `<g class="tick"><text x="${sx(v.x)}" y="${m.t + ih + 16}" text-anchor="middle">${v.label}</text></g>`;
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
      if (!finite(p.y) || (o.log && p.y <= 0)) {
        pen = false;
        continue;
      }
      d += (pen ? "L" : "M") + sx(p.x).toFixed(1) + "," + sy(p.y).toFixed(1);
      pen = true;
    }
    svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} clip-path="url(#${clipId})"/>`;
  }
  if (edge) {
    // a small triangle at the edge, labelled with the true value
    for (const s of o.series)
      for (const p of s.data) {
        if (!finite(p.y) || p.x < x0 || p.x > x1) continue;
        if (p.y >= edge[0] && p.y <= edge[1]) continue;
        const up = p.y > edge[1],
          cx = sx(p.x),
          cy = sy(p.y);
        const tip = up ? cy - 1 : cy + 1,
          base = up ? cy + 7 : cy - 7;
        svg += `<polygon class="edge-mark" points="${cx - 5},${base} ${cx + 5},${base} ${cx},${tip}" fill="${s.color}"/>`;
        svg += `<text class="axis-label edge-label" x="${cx + 7}" y="${up ? cy + 10 : cy - 3}" fill="${s.color}">${o.vfmt ? o.vfmt(p.y) : fmtNum(p.y)}</text>`;
      }
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
    let html = `<div style="margin-bottom:4px;font-weight:600">${xtip(best)}</div>`;
    for (const s of o.series) {
      const p = s.data.find(q => q.x === best);
      if (!p || !finite(p.y)) continue;
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
// show or hide a series on every chart
function toggleSeries(name) {
  if (HIDDEN.has(name)) HIDDEN.delete(name);
  else HIDDEN.add(name);
  schedule();
}
function legend(el, items, toggle = false) {
  el.innerHTML = items
    .map((i, k) => {
      const off = toggle && HIDDEN.has(i.name);
      const attrs = toggle
        ? ` data-i="${k}" class="toggle${off ? " off" : ""}" role="button" tabindex="0" aria-pressed="${!off}" title="Click to ${off ? "show" : "hide"}"`
        : "";
      return `<span${attrs}><i class="${i.dash ? "dash" : ""}" style="border-color:${i.color}"></i>${i.name}</span>`;
    })
    .join("");
  if (!toggle) return;
  for (const sp of el.querySelectorAll("span[data-i]")) {
    const name = items[+sp.dataset.i].name;
    sp.addEventListener("click", () => toggleSeries(name));
    sp.addEventListener("keydown", ev => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        toggleSeries(name);
      }
    });
  }
}

// ---------- percentages ----------
// "+1.5%" with a true minus sign, one decimal place
const pct = v =>
  v === null || !isFinite(v)
    ? "–"
    : (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(1) + "%";
// for prose: "about right (−0.3%)" below 0.5% in size
const nearZero = v => Math.abs(v) < 0.005;
const pctProse = v => (nearZero(v) ? `about right (${pct(v)})` : pct(v));
const pctList = a => (a.length ? a.map(pct).join(", ") : "–");
const meanShiftNote = m =>
  `Mean shift: the last ${m} days before the cut-off are not shown. A fixed shift cannot place any missing case there, so in practice these days would be left out.`;

// ---------- verdict ----------
function verdict(met) {
  const h = met.backsampling;
  const items = [];
  items.push(
    `Largest weekly error: <b>${pctProse(h.maxRel)}</b>. Largest Rt error <b>${(h.maxR >= 0 ? "+" : "−") + Math.abs(h.maxR).toFixed(2)}</b>. Weeks with Rt on the wrong side of 1: <b>${h.wrongSide}</b>.`
  );
  if (h.peakHeight !== null) {
    const ph = h.peakHeight;
    const size = nearZero(ph)
      ? `Peak height about right (${pct(ph)})`
      : `Peak ${ph < 0 ? "under" : "over"}stated by <b>${Math.abs(ph * 100).toFixed(1)}%</b>`;
    items.push(`${size}${h.peakShift ? `, moved ${h.peakShift} days` : ""}.`);
  }
  if (h.lastWeeks.length)
    items.push(
      `Last 3 weeks before the cut-off: <b>${pctList(h.lastWeeks)}</b> (mean shift ${pctList(met.mean.lastWeeks)}).`
    );
  document.getElementById("verdict").className = "verdict";
  document.getElementById("verdict").innerHTML =
    `<div><span class="tag">Back-sampling method.</span> ${items.join(" ")}</div>`;
}

// ---------- main render ----------
// results by parameter set, so the cut-off slider can replay quickly
const CACHE = new Map();
const CACHE_MAX = 40;
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
  const hit = CACHE.get(key);
  if (hit) {
    cache = hit;
    return cache;
  }
  const sc = scenario(P);
  const ex = expectedData(sc);
  const R = runMethods(sc, ex.comp, ex.miss, ex.empCounts);
  const W = Math.floor(Math.min(sc.N, sc.Td + 1) / 7);
  const st = P.stoch ? stochastic(sc, W, P.reps, P.nowcast && !P.retro) : null,
    stochMsg = "";
  cache = { key, sc, ex, R, W, st, stochMsg };
  if (CACHE.size >= CACHE_MAX) CACHE.delete(CACHE.keys().next().value);
  CACHE.set(key, cache);
  return cache;
}
function render() {
  const t0 = performance.now();
  const { sc, ex, R, W, st, stochMsg } = compute();
  const xw = w => sc.wk0 + 7 * w + 3.5;
  const off = sc.B + sc.wk0;
  const cs = comparisonSeries(P, sc, ex, R);
  const tw = weekly(cs.truth, W, off),
    rw = weekly(ex.rep, W, off);
  const mw = {};
  for (const M of METHODS)
    mw[M.key] = maskWeeks(
      weekly(cs.methods[M.key], W, off),
      sc,
      cs.lastDay[M.key]
    );
  const gT = growth(tw),
    gM = {};
  for (const M of METHODS) gM[M.key] = growth(mw[M.key]);
  const rIn = Array.from({ length: W }, (_, w) => {
    let s = 0;
    for (let d = 0; d < 7; d++) s += sc.r[off + 7 * w + d] ?? 0;
    return s / 7;
  });
  const col = {};
  for (const M of METHODS) col[M.key] = css(M.col);
  const ink = css("--text-primary"),
    grey = css("--reports");
  // weeks less than 80% (30% with the nowcast) reported by the cut-off
  const ok = completeWeeks(sc, W, cs.nowcast ? 0.3 : 0.8);
  const firstBad = ok.indexOf(false);
  const shades =
    firstBad >= 0
      ? [
          {
            from: sc.wk0 + 7 * firstBad,
            to: sc.wk0 + 7 * W,
            fill: "var(--grid)",
            label: cs.nowcast ? "under 30% reported" : "under 80% reported"
          }
        ]
      : [];
  const vline = P.retro ? undefined : sc.Td;
  // the x-axis covers the whole window, so it stays put as the cut-off moves
  const common = {
    shades,
    vline,
    dates: true,
    xtip: fmtWeek,
    xmin: ZOOM ? ZOOM[0] : 0,
    xmax: ZOOM ? ZOOM[1] : sc.N,
    onZoom: setZoom
  };
  const bandsFor = (src, map = b => b) =>
    METHODS.map(M => ({
      name: M.name,
      color: col[M.key],
      data: src[M.key].map(map)
    })).filter(shown);

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
    sA.map(s => ({ name: s.name, color: s.color, dash: !!s.dash })),
    true
  );
  lineChart(document.getElementById("chartA"), {
    ...common,
    series: sA.filter(shown),
    log: document.getElementById("logA").checked,
    xlab: "Onset or report week",
    ylab: "Cases per week",
    includeZero: true,
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
      y: tw[w] > 0 && finite(y) ? (y / tw[w] - 1) * 100 : null
    }))
  }));
  const bB = st
    ? bandsFor(st.bands, p => ({ x: p.x, lo: p.lo * 100, hi: p.hi * 100 }))
    : [];
  legend(
    document.getElementById("legendB"),
    sB.map(s => ({ name: s.name, color: s.color, dash: !!s.dash })),
    true
  );
  lineChart(document.getElementById("chartB"), {
    ...common,
    series: sB.filter(shown),
    bands: bB,
    zero: true,
    robust: 2,
    fixedRange: lockedRange("chartB"),
    xlab: "Onset week",
    ylab: "Error (%)",
    vfmt: v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1) + "%",
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
        { name: "True onsets", color: ink, data: toRBand(st.truthBand) },
        ...bandsFor(st.gbands).map(b => ({ ...b, data: toRBand(b.data) }))
      ].filter(shown)
    : [];
  legend(
    document.getElementById("legendC"),
    sC.map(s => ({ name: s.name, color: s.color, dash: !!s.dash })),
    true
  );
  lineChart(document.getElementById("chartC"), {
    ...common,
    series: sC.filter(shown),
    bands: bC,
    refY: 1,
    xlab: "Onset week",
    ylab: "Rt",
    yclamp: [0, 4],
    vfmt: v => v.toFixed(3),
    yfmt: v => v.toFixed(1),
    aria: "Reproduction number estimated from weekly counts"
  });

  // chart E: Rt error
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
  for (const M of METHODS) {
    if (HIDDEN.has(M.name)) continue;
    gM[M.key].forEach((y, w) => {
      if (y === null || gT[w] === null || !ok[w] || tw[w] < 10) return;
      if (wrongSideOf1(y, gT[w]))
        markE.push({ x: xw(w), y: rToR(y) - rToR(gT[w]), color: col[M.key] });
    });
  }
  const bE = st ? bandsFor(st.ebands) : [];
  legend(
    document.getElementById("legendE"),
    sE.map(s => ({ name: s.name, color: s.color, dash: !!s.dash })),
    true
  );
  lineChart(document.getElementById("chartE"), {
    ...common,
    series: sE.filter(shown),
    bands: bE,
    zero: true,
    robust: 0.02,
    fixedRange: lockedRange("chartE"),
    markers: markE,
    xlab: "Onset week",
    ylab: "Rt error",
    vfmt: v => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3),
    yfmt: v => v.toFixed(2),
    aria: "Error in Rt estimated from imputed counts"
  });

  // metrics
  const met = {};
  for (const M of METHODS)
    met[M.key] = metrics(tw, mw[M.key], gT, gM[M.key], ok, W, P.retro);
  verdict(met);
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
      ${P.retro ? "" : `<td>${pctList(q.lastWeeks)}</td>`}</tr>`;
  }
  html += `</tbody>`;
  document.getElementById("summary").innerHTML = html;
  const note = P.retro ? "" : meanShiftNote(cs.m);
  for (const id of ["meanNote", "meanNoteB"])
    document.getElementById(id).textContent = note;

  syncAsOf();
  drawSteps();

  const tot = tw.reduce((a, b) => a + b, 0);
  document.getElementById("status").textContent =
    `Expected cases in the onset weeks shown: ${Math.round(tot).toLocaleString("en-GB")}. ${P.retro ? "Retrospective: all cases reported" : `Cut-off ${fmtDate(sc.Td)} (day ${sc.Td})`}. Realised mean delay ${pmfMean(sc.fbar).toFixed(1)} days.${stochMsg} Rendered in ${(performance.now() - t0).toFixed(0)} ms.`;
}

// ---------- "real time as of" cut-off slider ----------
const ASOF = { timer: null };
const ASOF_MIN = 35;
function syncAsOf() {
  const el = document.getElementById("asofSel");
  const last = P.days - 1;
  const v = P.cutoff === null || P.cutoff === undefined ? last : P.cutoff;
  el.min = ASOF_MIN;
  el.max = last;
  el.value = Math.min(v, last);
  el.disabled = !!P.retro;
  document.getElementById("asofPlay").disabled = !!P.retro;
  document.getElementById("asofOut").textContent = P.retro
    ? "all cases reported (retrospective)"
    : fmtDate(Math.min(v, last)) + (v >= last ? " (last day shown)" : "");
}
// set the cut-off; at or past the last day shown it follows the window
function setCutoff(v) {
  P.cutoff = v >= P.days - 1 ? null : Math.max(v, 0);
  syncAsOf();
  schedule();
}
function stopPlay() {
  clearTimeout(ASOF.timer);
  ASOF.timer = null;
  document.getElementById("asofPlay").textContent = "▶ Play";
}
// one step of the playback: a week forward, stopping at the last day
function playStep() {
  const last = P.days - 1;
  const cur = P.cutoff ?? last;
  const next = cur + 7;
  setCutoff(next);
  if (next >= last) {
    stopPlay();
    return false;
  }
  ASOF.timer = setTimeout(playStep, 700);
  return true;
}
function togglePlay() {
  if (ASOF.timer) {
    stopPlay();
    return;
  }
  if (P.retro) return;
  if (P.cutoff === null || P.cutoff === undefined) setCutoff(ASOF_MIN);
  document.getElementById("asofPlay").textContent = "❚❚ Pause";
  ASOF.timer = setTimeout(playStep, 700);
}
function wireAsOf() {
  const el = document.getElementById("asofSel");
  el.addEventListener("input", () => {
    stopPlay();
    setCutoff(+el.value);
  });
  document.getElementById("asofPlay").addEventListener("click", togglePlay);
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
      stopPlay();
      activePreset = i;
      ZOOM = null;
      document.getElementById("zoomReset").disabled = true;
      P = fromPreset(p.set);
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
      } else if (c.type === "date") {
        div.innerHTML = `<label for="c_${c.id}">${c.label}</label><input type="date" id="c_${c.id}" required>`;
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
        if (c.type === "date") {
          // dates only relabel the charts; the scenario is unchanged
          if (parseStart(el.value) !== null) START = el.value;
          syncControls();
          schedule();
          return;
        }
        P[c.id] = c.type === "check" ? el.checked : +el.value;
        if (c.id === "retro" && P.retro) stopPlay();
        // a cut-off at or past the last day shown follows the window
        if (c.id === "days" && P.cutoff !== null && P.cutoff >= P.days - 1)
          P.cutoff = null;
        if (c.id === "days") syncControls();
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
      else if (c.type === "date") el.value = START;
      else {
        el.value = P[c.id];
        document.getElementById("o_" + c.id).textContent = c.fmt(P[c.id]);
      }
    }
  syncEnabled();
  syncAsOf();
  drawEditor();
  markPreset();
}
// ---------- curve editors ----------
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
    show: v => v.toFixed(3),
    name: "r"
  },
  {
    el: "muEditor",
    key: "muKnots",
    min: 0,
    max: 20,
    step: 0.5,
    ref: null,
    grid: [0, 5, 10, 15, 20],
    tick: v => v.toFixed(0),
    show: v => v.toFixed(1) + " days",
    name: "mean delay"
  },
  {
    el: "piEditor",
    key: "piKnots",
    min: 0,
    max: 1,
    step: 0.01,
    ref: null,
    grid: [0, 0.25, 0.5, 0.75, 1],
    tick: v => Math.round(v * 100) + "%",
    show: v => Math.round(v * 100) + "%",
    name: "missing"
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
    show: v => v.toFixed(2),
    name: "ascertainment"
  }
];
// knot values for an editor; presets that set mu0/mu1 or pi0/pi1 give a
// straight line over the days shown
function curveVals(key) {
  if (P[key]) return P[key];
  if (key === "muKnots")
    return knotsFrom([
      [0, P.mu0],
      [P.days - 1, P.mu1]
    ]);
  if (key === "piKnots")
    return knotsFrom([
      [0, P.pi0],
      [P.days - 1, P.pi1]
    ]);
  return KNOTS.map(() => 0);
}
const ED = { w: 296, h: 150, l: 34, r: 8, t: 8, b: 20 };
// the editors span only the days shown: the last knot at or after the
// final day of the window
function edEnd() {
  return Math.min(KNOT_END, Math.ceil((P.days - 1) / 15) * 15);
}
function edScales(cfg) {
  const iw = ED.w - ED.l - ED.r,
    ih = ED.h - ED.t - ED.b,
    end = edEnd(),
    nK = end / 15 + 1;
  return {
    sx: t => ED.l + (t / end) * iw,
    sy: v => ED.t + ((cfg.max - v) / (cfg.max - cfg.min)) * ih,
    ix: px => clamp(Math.round(((px - ED.l) / iw) * (nK - 1)), 0, nK - 1),
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
      vals = curveVals(cfg.key);
    let g = `<svg viewBox="0 0 ${ED.w} ${ED.h}" style="width:100%;height:auto;touch-action:none;cursor:crosshair">`;
    for (const v of cfg.grid) {
      g += `<line x1="${ED.l}" x2="${ED.w - ED.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(${v === cfg.ref ? "--text-muted" : "--grid"})"/>`;
      g += `<text x="${ED.l - 4}" y="${sy(v) + 3}" text-anchor="end" font-size="9" fill="var(--text-muted)">${cfg.tick(v)}</text>`;
    }
    const end = edEnd(),
      shown = KNOTS.filter(t => t <= end);
    for (let t = 0; t <= end; t += end <= 180 ? 30 : 60)
      g += `<text x="${sx(t)}" y="${ED.h - 6}" text-anchor="middle" font-size="9" fill="var(--text-muted)">${t}</text>`;
    g += `<polyline fill="none" stroke="var(--text-primary)" stroke-width="2" points="${shown.map((t, j) => `${sx(t)},${sy(vals[j])}`).join(" ")}"/>`;
    shown.forEach((t, j) => {
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
      if (!P[cfg.key]) P[cfg.key] = curveVals(cfg.key).slice();
      P[cfg.key][j] = +v.toFixed(3);
      activePreset = -1;
      markPreset();
      drawEditor();
      el.querySelector(".edVal").textContent =
        `day ${KNOTS[j]} (${fmtShort(KNOTS[j])}): ${cfg.name} ${cfg.show(P[cfg.key][j])}`;
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
  document.getElementById("c_ended").disabled = !P.retro;
  document.getElementById("c_nowcast").disabled = P.retro;
}
let timer = null;
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(render, 40);
}
document.getElementById("tSel").addEventListener("input", () => {
  stopRun();
  schedule();
});
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
document.getElementById("lockAxes").addEventListener("change", () => {
  for (const k of Object.keys(LOCKED)) delete LOCKED[k];
  schedule();
});
document.getElementById("themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur
    ? cur === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  schedule();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);

// Follow the docs' light or dark choice. Inside the docs preview this is
// the parent page's scheme; otherwise it is the palette the docs saved.
function docsScheme() {
  try {
    if (window.parent && window.parent !== window)
      return window.parent.document.body.getAttribute("data-md-color-scheme");
    const key = new URL("docs/", location.href).pathname + ".__palette";
    const saved = JSON.parse(localStorage.getItem(key) || "null");
    return saved && saved.color ? saved.color.scheme : null;
  } catch {
    return null;
  }
}
function followDocs() {
  const scheme = docsScheme();
  if (!scheme) return;
  const theme = scheme === "slate" ? "dark" : "light";
  if (document.documentElement.getAttribute("data-theme") === theme) return;
  document.documentElement.setAttribute("data-theme", theme);
  schedule();
}
followDocs();
try {
  if (window.parent && window.parent !== window) {
    document.documentElement.classList.add("embedded");
    new MutationObserver(followDocs).observe(window.parent.document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  }
} catch {
  // a parent on another origin: keep the page's own theme
}
let lastWidth = 0;
new ResizeObserver(entries => {
  const w = Math.round(entries[0].contentRect.width);
  if (w !== lastWidth) {
    lastWidth = w;
    schedule();
  }
}).observe(document.querySelector(".charts"));

// ---------- step-through of the methods on the current scenario ----------
const STEP = { method: "backsampling", step: 0, iter: 40, timer: null };
const STEP_TITLES = [
  "What is observed",
  "The delay the method uses",
  "Where the onset could be",
  "Impute the missing onsets",
  "Apply to every report day"
];
const NSTEP = STEP_TITLES.length;
// days shown before the report day in the last step
const STEP_WINDOW = 60;
// the data as of report day t: the same scenario cut at t, in real time.
// Cached by parameters and t.
const VINTAGES = new Map();
function vintage(t) {
  const p = { ...P, retro: false, cutoff: t, stoch: false };
  const key = JSON.stringify(p);
  let v = VINTAGES.get(key);
  if (v) return v;
  const sc = scenario(p);
  const ex = expectedData(sc);
  const R = runMethods(sc, ex.comp, ex.miss, ex.empCounts);
  const cs = comparisonSeries(p, sc, ex, R);
  v = { key, p, sc, ex, R, cs, snaps: null };
  if (VINTAGES.size >= 30) VINTAGES.delete(VINTAGES.keys().next().value);
  VINTAGES.set(key, v);
  return v;
}
// EM iterations on a vintage from a start using only the complete cases,
// for the animation
function emSnapshots(v) {
  if (v.snaps) return v.snaps;
  const snaps = [];
  emImpute(v.ex.comp, v.ex.miss, v.sc, v.ex.comp, 40, (it, out) =>
    snaps.push(out)
  );
  v.snaps = snaps;
  return snaps;
}
// the onset curve λ(s) the generative method ends with: its imputed
// onsets corrected for right truncation
const lambdaOf = v =>
  Float64Array.from(v.R.gen, (x, s) => x / Math.max(v.sc.cover[s] ?? 1, 1e-4));
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
  const day = v.date;
  if (step === 0)
    return `A sample of cases from this scenario, using the data as of ${day}. Each has an onset (dot) and a report day (square). Some have no onset date (?). On ${day}, about ${((100 * v.missT) / Math.max(v.repT, 1e-9)).toFixed(0)}% of cases reported have no onset date. The three at the bottom are the ones we follow.`;
  if (step === 1) {
    if (key === "backsampling")
      return `Back-sampling pools the delays of every case with both dates in the data so far, mean ${v.gMean.toFixed(1)} days. Ink is the true delay for missing cases reported on ${day}, mean ${v.bMean.toFixed(1)} days. It depends on how onsets were changing before then. The pooled delay is the same whatever day a case was reported.`;
    if (key === "mean")
      return `The mean shift keeps only the mean of the pooled delays, ${v.gMean.toFixed(1)} days, rounded to ${v.m}. Ink is the true delay for missing cases reported on ${day}, mean ${v.bMean.toFixed(1)} days.`;
    if (key === "fwd")
      return `The forward draw uses the delay from onset to report, mean ${v.fMean.toFixed(1)} days, as estimated with corrections for truncation and censoring. Ink is the true delay for missing cases reported on ${day}, mean ${v.bMean.toFixed(1)} days. The forward delay ignores how many onsets there were on each earlier day.`;
    return `Bars are the forward delay f(t − s): the chance that a case with onset on day s is reported on ${day}. The dashed line is the method's estimate of the onset curve, λ(s), on the same days. Ink is the true delay for missing cases reported on ${day}, mean ${v.bMean.toFixed(1)} days.`;
  }
  if (step === 2) {
    if (key === "mean")
      return `Every missing case reported on ${day} goes ${v.m} days back, to ${v.shiftDate}. Ink is where such cases truly had their onsets.`;
    if (key === "gen")
      return `Each onset day s gets weight λ(s) × f(t − s): the estimated onsets that day (dashed line) times the chance of that delay (faint bars). Solid bars are the product, scaled to sum to 1. Days with more onsets pull weight towards them, so a rise or fall in onsets moves the weights away from the delay alone. Ink is where such cases truly had their onsets.`;
    return `${nm} gives each onset day the probability of the matching delay. Ink is where such cases truly had their onsets.`;
  }
  if (step === 3) {
    if (key === "backsampling" || key === "fwd")
      return `Each missing case gets one random delay from the distribution, so one imputed onset (filled dot). The ring is its true onset, which real data would not show.`;
    if (key === "mean")
      return `Each missing case moves back exactly ${v.m} days (filled dot). The ring is its true onset, which real data would not show.`;
    return `Each missing case is shared across onset days in proportion to the weights (dots sized by share). The ring is its true onset, which real data would not show.`;
  }
  const target = v.nowcast
    ? "all onsets that will eventually be reported. Both are corrected for right truncation"
    : "the onsets of cases reported by then";
  const meanNote =
    key === "mean"
      ? ` The last ${v.m} days are not shown: a fixed shift cannot place any case there.`
      : "";
  if (key === "gen")
    return `Doing this for every report day gives the imputed onset curve, using the data as of ${day}, against ${target}. The generative method then re-estimates the onset curve and repeats. Iteration ${v.iter} of 40: press Run to watch it converge from a start using the complete cases only.`;
  return `Doing this for every report day gives the imputed onset curve, using the data as of ${day}, against ${target}.${meanNote}`;
}
function drawCases(el, o) {
  const { cases, t, key, step, color, weights, truthW, dist, m, lam } = o;
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
    g += `<text x="${X(d)}" y="${axisY + 13}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${fmtShort(d)}</text>`;
  }
  cases.forEach((cs, i) => {
    const y = top + i * rowH + 4,
      faded = step >= 2 && !cs.focus;
    const op = faded ? 0.3 : 1;
    if (!cs.missing) {
      const hl = step === 1 && (key === "backsampling" || key === "mean");
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
    // generative, steps 2 and 3: f(t − s) as bars, λ(s) as a line; in
    // step 3 the faint bars are f and the solid bars the product
    const genLayers = key === "gen" && step <= 2;
    const solid = step === 1 ? dist : weights;
    const faint = key === "gen" && step === 2 ? dist : null;
    const maxW = Math.max(
      ...solid.slice(0, 33),
      ...(faint ? faint.slice(0, 33) : []),
      ...truthW.slice(0, 33),
      0.01
    );
    const bh = w => ((lowBase - lowTop) * w) / maxW;
    const label =
      step === 1 ? "delay back from " + fmtShort(t) : "P(onset day)";
    if (key !== "mean") {
      if (faint)
        for (let d = 0; d <= 32; d++)
          g += `<rect x="${X(t - d) - 7}" y="${lowBase - bh(faint[d] || 0)}" width="14" height="${bh(faint[d] || 0)}" fill="${color}" opacity="0.22"/>`;
      for (let d = 0; d <= 32; d++)
        g += `<rect x="${X(t - d) - (faint ? 4 : 6)}" y="${lowBase - bh(solid[d] || 0)}" width="${faint ? 8 : 12}" height="${bh(solid[d] || 0)}" fill="${color}" opacity="${faint ? 0.9 : 0.75}"/>`;
    }
    if (key === "mean")
      g += `<line x1="${X(t - m)}" x2="${X(t - m)}" y1="${lowTop}" y2="${lowBase}" stroke="${color}" stroke-width="4"/>`;
    if (genLayers && lam) {
      // λ(s) on its own scale, over the same days
      let lmax = 0;
      for (let d = 0; d <= 32; d++) lmax = Math.max(lmax, lam[d] || 0);
      const ly = x => lowBase - ((lowBase - lowTop) * x) / (lmax || 1);
      const lp = [];
      for (let d = 32; d >= 0; d--) lp.push(`${X(t - d)},${ly(lam[d] || 0)}`);
      g += `<polyline class="lambda" points="${lp.join(" ")}" fill="none" stroke="var(--text-secondary)" stroke-width="2" stroke-dasharray="6 3"/>`;
      g += `<text x="${X(from) + 4}" y="${ly(lam[32] || 0) - 6}" font-size="11" fill="var(--text-secondary)">λ(s): estimated onsets per day</text>`;
    }
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
  const key = STEP.method,
    step = STEP.step;
  const M = METHODS.find(x => x.key === key),
    color = css(M.col),
    ink = css("--text-primary"),
    grey = css("--reports");
  const tSel = document.getElementById("tSel");
  const tMax = P.days - 1;
  tSel.max = tMax;
  if (+tSel.value > tMax) tSel.value = tMax;
  const t = clamp(+tSel.value || 0, 0, tMax);
  document.getElementById("tSelOut").textContent = fmtDate(t);
  // every step uses the data as of day t
  const vt = vintage(t);
  const { sc, ex, R } = vt;
  const ti = t + sc.B;
  const back = new Float64Array(D + 1);
  for (let d = 0; d <= D; d++) {
    const s = ti - d;
    if (s >= 0 && s < sc.S) back[d] = sc.C[s] * sc.f[s][d] * sc.pim(ti, d);
  }
  const bn = normalise(back);
  const m = vt.cs.m;
  // the delay each method uses, indexed by days back from t; for the
  // generative method f(t − s) for each onset day s
  const fts = new Float64Array(D + 1);
  for (let d = 0; d <= D; d++) {
    const s = ti - d;
    if (s >= 0 && s < sc.S) fts[d] = sc.f[s][d];
  }
  const dist =
    key === "backsampling" || key === "mean"
      ? R.g
      : key === "fwd"
        ? sc.fbar
        : fts;
  const lamAll = lambdaOf(vt);
  const lam = new Float64Array(D + 1);
  const w = new Float64Array(D + 1);
  for (let d = 0; d <= D; d++) {
    const s = ti - d;
    if (s < 0 || s >= sc.S) continue;
    lam[d] = lamAll[s];
    if (key === "backsampling") w[d] = R.g[d];
    else if (key === "fwd") w[d] = sc.fbar[d];
    else if (key === "gen") w[d] = lamAll[s] * fts[d];
  }
  const weights = normalise(w);
  const v = {
    t,
    date: fmtDate(t),
    shiftDate: fmtDate(t - m),
    missT: ex.miss[ti] || 0,
    repT: ex.rep[ti] || 0,
    gMean: pmfMean(R.g),
    fMean: pmfMean(sc.fbar),
    bMean: pmfMean(bn),
    m,
    iter: STEP.iter,
    nowcast: vt.cs.nowcast
  };
  const legendEl = document.getElementById("stepLegend");
  if (step < 4) {
    const cases = sampleCases(sc, t, bn);
    const items = [
      { name: "Truth for " + fmtShort(t), color: ink },
      { name: M.name, color }
    ];
    if (key === "gen" && (step === 1 || step === 2))
      items.push({ name: "λ(s), onset curve", color: grey, dash: true });
    legend(legendEl, items);
    drawCases(el, {
      cases,
      t,
      key,
      step,
      color,
      weights: Array.from(weights),
      truthW: Array.from(bn),
      dist: Array.from(dist),
      lam: Array.from(lam),
      m
    });
  } else {
    const from = Math.max(0, t - STEP_WINDOW);
    const day = (arr, last = Infinity) => {
      const out = [];
      for (let x = from; x <= t; x++)
        out.push({ x, y: x > last ? null : (arr[x + sc.B] ?? null) });
      return out;
    };
    let imputed = vt.cs.methods[key];
    if (key === "gen") {
      const snaps = emSnapshots(vt);
      const snap = snaps[Math.min(STEP.iter, snaps.length) - 1];
      imputed = vt.cs.nowcast ? correctTruncation(snap, sc) : snap;
    }
    const last = vt.cs.lastDay[key] ?? Infinity;
    const series = [
      { name: "Truth", color: ink, width: 3, data: day(vt.cs.truth) },
      {
        name: key === "gen" ? `${M.name}, iteration ${STEP.iter}` : M.name,
        color,
        width: 2.5,
        data: day(imputed, last)
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
      xmin: from,
      xmax: t,
      dates: true,
      includeZero: true,
      height: 260,
      vlines: [{ x: t, color: grey, label: `data as of ${fmtShort(t)}` }],
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
wireAsOf();
render();
