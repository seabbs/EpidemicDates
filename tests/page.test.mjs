// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Runs the page script in Node against a stub DOM, to catch undefined
// names and runtime errors in page.js without a browser.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const pageFile = fileURLToPath(new URL("../page.js", import.meta.url));
const page = readFileSync(pageFile, "utf8");
const EpidemicDates = createRequire(import.meta.url)("../epidemic-dates.js");

// an element stub that accepts any property access, call or assignment
const stub = () =>
  new Proxy(function () {}, {
    get: (_, k) => {
      if (k === "clientWidth") return 600;
      if (k === "value") return "90";
      if (k === "checked") return false;
      if (k === "dataset") return {};
      if (k === Symbol.toPrimitive) return () => 0;
      if (k === "querySelectorAll") return () => [];
      return stub();
    },
    set: () => true,
    apply: () => stub(),
    construct: () => stub()
  });

function loadPage() {
  const ctx = {
    EpidemicDates,
    Math,
    JSON,
    Array,
    Float64Array,
    Number,
    Set,
    Map,
    Object,
    String,
    console,
    document: {
      getElementById: () => stub(),
      createElement: () => stub(),
      createElementNS: () => stub(),
      querySelector: () => stub(),
      querySelectorAll: () => [],
      documentElement: stub()
    },
    getComputedStyle: () => ({ getPropertyValue: () => "#000" }),
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    ResizeObserver: class {
      observe() {}
    },
    window: { addEventListener() {}, removeEventListener() {} },
    performance: { now: () => 0 },
    setTimeout: () => 0,
    clearTimeout() {}
  };
  vm.createContext(ctx);
  // the file name lets V8 coverage count page.js
  vm.runInContext(page, ctx, { filename: pageFile });
  return ctx;
}

test("page script loads and renders the default scenario", () => {
  const ctx = loadPage();
  assert.doesNotThrow(() => vm.runInContext("render()", ctx));
});

test("every preset renders", () => {
  const ctx = loadPage();
  vm.runInContext(
    `for (const pr of PRESETS) {
    P = { ...DEFAULTS, ...pr.set };
    P.knots = P.knots.slice(); P.asc = P.asc.slice();
    render();
  }`,
    ctx
  );
});

test("stochastic replicates render", () => {
  const ctx = loadPage();
  vm.runInContext("P.stoch = true; P.reps = 5; render();", ctx);
});

test("every step of the step-through renders for every method", () => {
  const ctx = loadPage();
  vm.runInContext(
    `for (const m of METHODS) for (let s = 0; s < NSTEP; s++) {
    STEP.method = m.key; STEP.step = s; drawSteps();
  }
  STEP.method = "gen"; STEP.step = NSTEP - 1;
  for (let i = 1; i <= 40; i++) { STEP.iter = i; drawSteps(); }`,
    ctx
  );
});

test("editing the curves does not change the module's constants", () => {
  const ctx = loadPage();
  const k0 = EpidemicDates.DEFAULTS.knots[0];
  const a0 = EpidemicDates.PRESETS[0].set.asc[0];
  const m0 = EpidemicDates.PRESETS[0].set.muKnots[0];
  vm.runInContext(
    "P.knots[0] = 0.1; P.asc[0] = 2; P.muKnots[0] = 3; P.piKnots[0] = 0.5;" +
      " P.seed = 3; render();",
    ctx
  );
  assert.equal(EpidemicDates.DEFAULTS.knots[0], k0);
  assert.equal(EpidemicDates.PRESETS[0].set.asc[0], a0);
  assert.equal(EpidemicDates.PRESETS[0].set.muKnots[0], m0);
});

test("stochastic replicates on the page are seeded", () => {
  const ctx = loadPage();
  const bands = () =>
    vm.runInContext(
      "cache = { key: null }; P.stoch = true; P.reps = 5; " +
        "JSON.stringify(compute().st)",
      ctx
    );
  assert.equal(bands(), bands());
});

test("every preset renders when loaded as the page loads it", () => {
  const ctx = loadPage();
  vm.runInContext(
    `for (const pr of PRESETS) {
    P = fromPreset(pr.set); syncControls(); render();
  }`,
    ctx
  );
});

test("the cut-off slider sets the real-time cut-off and plays by week", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  assert.equal(run("P.cutoff"), null);
  run("setCutoff(100); render();");
  assert.equal(run("P.cutoff"), 100);
  assert.equal(run("compute().sc.Td"), 100);
  // play steps forward a week at a time and stops at the last day
  assert.equal(run("playStep()"), true);
  assert.equal(run("P.cutoff"), 107);
  run("setCutoff(P.days - 3); playStep();");
  assert.equal(run("P.cutoff"), null);
  run("togglePlay(); render();");
  assert.equal(run("P.cutoff"), run("ASOF_MIN"));
  run("stopPlay(); setCutoff(9999);");
  assert.equal(run("P.cutoff"), null);
});

test("legend entries hide series on every chart and Input starts hidden", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  assert.equal(run("HIDDEN.has('Input')"), true);
  run("toggleSeries('Input'); toggleSeries(METHODS[0].name); render();");
  assert.equal(run("HIDDEN.has('Input')"), false);
  assert.equal(run("HIDDEN.has(METHODS[0].name)"), true);
  // the state survives a redraw and a new preset
  run("P = fromPreset(PRESETS[2].set); render();");
  assert.equal(run("HIDDEN.has(METHODS[0].name)"), true);
});

test("dates label the charts from the start date", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  assert.equal(run("fmtDate(174)"), "24 Oct 2026");
  assert.equal(run("fmtWeek(3.5)"), "Week of 3 May to 9 May 2026");
  assert.equal(run("parseStart('not a date')"), null);
  const ticks = run("JSON.stringify(dateTicks(0, 174, 8))");
  assert.deepEqual(
    JSON.parse(ticks).map(t => t.label),
    ["Jun 2026", "Jul", "Aug", "Sep", "Oct"]
  );
  run("START = '2026-01-01'; render();");
  assert.equal(run("fmtDate(0)"), "1 Jan 2026");
});

test("error charts get a minimum range and a robust cap", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  assert.equal(run("JSON.stringify(robustRange([1, -2, 3], 10))"), "[-10,10]");
  // one extreme value does not set the range
  const r = JSON.parse(
    run(
      "JSON.stringify(robustRange([...Array.from({length: 40}, (_, i) => i % 7 - 3), 500], 0.2))"
    )
  );
  assert.ok(r[1] < 50 && r[1] >= 3 && r[0] <= -3);
  // the extreme point is drawn at the edge with its value
  const html = run(`(() => {
    const el = { clientWidth: 600, innerHTML: "",
      querySelector: () => document.getElementById("x") };
    lineChart(el, { robust: 10, xlab: "", ylab: "", vfmt: v => v + "!",
      series: [{ name: "a", color: "red",
        data: Array.from({ length: 40 }, (_, i) =>
          ({ x: i, y: i === 20 ? 900 : i % 5 })) }] });
    return el.innerHTML;
  })()`);
  assert.match(html, /edge-mark/);
  assert.match(html, />900!</);
});

test("the last step uses the data as of the report day, cached", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  run("P.retro = true; render();");
  assert.equal(run("vintage(120) === vintage(120)"), true);
  assert.equal(run("vintage(120).sc.Td"), 120);
  assert.equal(run("vintage(120).p.retro"), false);
  assert.equal(
    run("vintage(120).cs.lastDay.mean"),
    run("120 - vintage(120).R.m")
  );
  // the animation runs on the same vintage
  run(
    "STEP.method = 'gen'; STEP.step = NSTEP - 1; STEP.iter = 5; drawSteps();"
  );
  assert.equal(
    run("vintage(+document.getElementById('tSel').value).snaps.length"),
    40
  );
  assert.match(run("stepText('gen', 2, { date: 'x', m: 7 })"), /pull weight/);
});

test("bulk edits change a whole curve and reset restores the preset", () => {
  const ctx = loadPage();
  const run = code => vm.runInContext(code, ctx);
  const cfg = 'EDITORS.find(e => e.key === "muKnots")';
  const before = JSON.parse(run(`JSON.stringify(curveVals("muKnots"))`));
  run(`bulkEdit(${cfg}, "add", 2)`);
  const after = JSON.parse(run(`JSON.stringify(curveVals("muKnots"))`));
  after.forEach((v, j) =>
    assert.ok(Math.abs(v - Math.min(before[j] + 2, 20)) < 0.26)
  );
  run(`bulkEdit(${cfg}, "set", 99)`);
  assert.ok(
    JSON.parse(run(`JSON.stringify(curveVals("muKnots"))`)).every(v => v === 20)
  );
  run(`resetCurve(${cfg})`);
  assert.deepEqual(
    JSON.parse(run(`JSON.stringify(curveVals("muKnots"))`)),
    before
  );
});
