// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Runs the page script in Node against a stub DOM, to catch undefined
// names and runtime errors in page.js without a browser.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";

const page = readFileSync(new URL("../page.js", import.meta.url), "utf8");
const OnsetImputation = createRequire(import.meta.url)("../onset-imputation.js");

// an element stub that accepts any property access, call or assignment
const stub = () => new Proxy(function () {}, {
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
    OnsetImputation, Math, JSON, Array, Float64Array, Number, Set, Map,
    Object, String, console,
    document: {
      getElementById: () => stub(), createElement: () => stub(),
      createElementNS: () => stub(), querySelector: () => stub(),
      querySelectorAll: () => [], documentElement: stub()
    },
    getComputedStyle: () => ({ getPropertyValue: () => "#000" }),
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    ResizeObserver: class { observe() {} },
    window: { addEventListener() {}, removeEventListener() {} },
    performance: { now: () => 0 },
    setTimeout: () => 0, clearTimeout() {}
  };
  vm.createContext(ctx);
  vm.runInContext(page, ctx);
  return ctx;
}

test("page script loads and renders the default scenario", () => {
  const ctx = loadPage();
  assert.doesNotThrow(() => vm.runInContext("render()", ctx));
});

test("every preset renders", () => {
  const ctx = loadPage();
  vm.runInContext(`for (const pr of PRESETS) {
    P = { ...DEFAULTS, ...pr.set };
    P.knots = P.knots.slice(); P.asc = P.asc.slice();
    render();
  }`, ctx);
});

test("stochastic replicates render", () => {
  const ctx = loadPage();
  vm.runInContext("P.stoch = true; P.reps = 5; render();", ctx);
});

test("every step of the step-through renders for every method", () => {
  const ctx = loadPage();
  vm.runInContext(`for (const m of METHODS) for (let s = 0; s < 4; s++) {
    STEP.method = m.key; STEP.step = s; drawSteps();
  }`, ctx);
});
