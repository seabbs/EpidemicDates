// Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
// Turns the "Packages" tab into a dropdown of the four language guides.
(function () {
  const LANGS = [
    ["Python", "python/"],
    ["JavaScript", "javascript/"],
    ["R", "r/"],
    ["Julia", "julia/"]
  ];
  function setup() {
    const tabs = document.querySelectorAll(".md-tabs__item");
    const item = [...tabs].find(
      t => t.textContent.trim() === "Packages"
    );
    if (!item || item.querySelector(".packages-menu")) return;
    // the tab's own link points at the first guide, so build links from it
    const link = item.querySelector("a");
    const base = new URL(link.getAttribute("href"), location.href);
    const root = new URL("../", base);
    const menu = document.createElement("div");
    menu.className = "packages-menu";
    menu.setAttribute("role", "menu");
    for (const [name, path] of LANGS) {
      const a = document.createElement("a");
      a.href = new URL(path, root).href;
      a.textContent = name;
      a.setAttribute("role", "menuitem");
      menu.appendChild(a);
    }
    item.classList.add("has-packages-menu");
    item.appendChild(menu);
    link.setAttribute("aria-haspopup", "true");
  }
  // MkDocs Material swaps pages without a reload when instant loading is on
  if (window.document$ && window.document$.subscribe)
    window.document$.subscribe(setup);
  else document.addEventListener("DOMContentLoaded", setup);
})();
