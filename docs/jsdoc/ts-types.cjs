// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
/**
 * JSDoc plugin that rewrites TypeScript-only type syntax into types the
 * JSDoc (Closure) parser accepts, so jsdoc-to-markdown can read
 * `onset-imputation.js`. It only changes the docs, not the source.
 *
 * - Function types such as `(a: T) => U` become `function`.
 * - `readonly` modifiers are dropped.
 * - Tuples such as `[number, number]` become `Array`.
 *
 * Bare typedef names, such as `Scenario`, are qualified with the module
 * name so the reference links to them.
 *
 * Links to the old JSDoc home page (`index.html`) point to the JavaScript
 * guide (`index.md`) instead.
 *
 * It also hides inner helpers (doclets with `inner` scope), so the
 * reference lists only the public API.
 */
"use strict";

const TAG =
  /@(?:param|arg|argument|type|returns?|property|prop|typedef|throws)\s*\{/g;

let moduleName = null;
let typedefs = [];

/** Find the module name and typedef names in a source file. */
function collect(source) {
  const mod = /@module\s+(\S+)/.exec(source);
  if (mod) moduleName = mod[1];
  typedefs = [
    ...source.matchAll(
      /@(?:typedef\s*\{[^}]*\}|callback)\s*([A-Za-z_$][\w$]*)/g
    )
  ].map(m => m[1]);
}

/** Qualify bare typedef names with the module name. */
function qualify(type) {
  if (!moduleName || typedefs.length === 0) return type;
  const names = new RegExp(`(^|[^\\w$.:])(${typedefs.join("|")})\\b`, "g");
  return type.replace(names, `$1module:${moduleName}.$2`);
}

/** Simplify one type expression. */
function simplify(type) {
  if (type.includes("=>")) return "function";
  let out = qualify(type.replace(/\breadonly\s+/g, ""));
  let prev;
  do {
    prev = out;
    out = out.replace(/\[[^[\]]*\]/g, m => (m === "[]" ? m : "Array"));
  } while (out !== prev);
  return out;
}

/** Rewrite every `{type}` that follows a tag in a doc comment. */
function rewrite(comment) {
  let out = "";
  let last = 0;
  TAG.lastIndex = 0;
  let m;
  while ((m = TAG.exec(comment)) !== null) {
    const start = m.index + m[0].length;
    let depth = 1;
    let i = start;
    for (; i < comment.length && depth > 0; i++) {
      if (comment[i] === "{") depth++;
      else if (comment[i] === "}") depth--;
    }
    const end = i - 1;
    const type = comment.slice(start, end).replace(/\s*\n\s*\*\s*/g, " ");
    out += comment.slice(last, start) + simplify(type);
    last = end;
    TAG.lastIndex = end;
  }
  return out + comment.slice(last);
}

exports.handlers = {
  beforeParse(e) {
    collect(e.source);
  },
  jsdocCommentFound(e) {
    e.comment = rewrite(e.comment).replace(/\]\(index\.html\)/g, "](index.md)");
  },
  newDoclet(e) {
    if (e.doclet.scope === "inner") e.doclet.ignore = true;
  }
};
