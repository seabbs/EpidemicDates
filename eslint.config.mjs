// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// ESLint flat config: the recommended rules plus a few stricter ones.
// eslint-config-prettier turns off rules that clash with prettier.
import js from "@eslint/js";
import prettier from "eslint-config-prettier";
import globals from "globals";

const rules = {
  "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }],
  "no-var": "error",
  "prefer-const": "error",
  eqeqeq: ["error", "smart"]
};

export default [
  {
    ignores: ["_site/**", ".venv/**", "node_modules/**", "docs/**"]
  },
  js.configs.recommended,
  {
    files: ["epidemic-dates.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: { module: "writable" }
    },
    rules
  },
  {
    files: ["page.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: { ...globals.browser, EpidemicDates: "readonly" }
    },
    rules
  },
  {
    files: ["**/*.mjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: { ...globals.node }
    },
    rules
  },
  prettier
];
