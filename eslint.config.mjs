// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// ESLint flat config. No plugins, so `npx eslint@10 .` needs nothing installed.
const browser = {
  window: "readonly", document: "readonly", getComputedStyle: "readonly",
  matchMedia: "readonly", ResizeObserver: "readonly", performance: "readonly",
  setTimeout: "readonly", clearTimeout: "readonly", console: "readonly"
};
const node = {
  require: "readonly", module: "writable", process: "readonly",
  console: "readonly", globalThis: "readonly", URL: "readonly"
};
const rules = {
  "no-undef": "error",
  "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }],
  "no-redeclare": "error",
  "no-dupe-keys": "error",
  "no-unreachable": "error",
  "no-var": "error",
  "prefer-const": "error",
  eqeqeq: ["error", "smart"]
};

export default [
  { ignores: ["_site/**", ".venv/**", "node_modules/**"] },
  {
    files: ["onset-imputation.js"],
    languageOptions: { ecmaVersion: 2022, sourceType: "script", globals: { ...node } },
    rules
  },
  {
    files: ["page.js"],
    languageOptions: {
      ecmaVersion: 2022, sourceType: "script",
      globals: { ...browser, OnsetImputation: "readonly" }
    },
    rules
  },
  {
    files: ["**/*.mjs"],
    languageOptions: { ecmaVersion: 2022, sourceType: "module", globals: { ...node } },
    rules
  }
];
