// Tiny i18n: English is written in the HTML, Russian comes from dictionaries
// (assets/ru-common.js and topics/<topic>.ru.js). Switching swaps text in place,
// with no page reload.
//
// - <el data-i18n="key"> gets dict[key]; <el data-i18n> uses its own English text as the key.
// - Scripts call I18N.t("English {n}", { n: 3 }) for text they build themselves,
//   and I18N.onChange(fn) to redraw when the language flips.

(function () {
  "use strict";

  const dict = {};
  const listeners = [];
  let toggle = null;

  let lang = "";
  try { lang = localStorage.getItem("lang") || ""; } catch (e) { /* storage blocked */ }
  if (lang !== "en" && lang !== "ru") {
    lang = (navigator.language || "").toLowerCase().indexOf("ru") === 0 ? "ru" : "en";
  }

  const norm = (s) => s.replace(/\s+/g, " ").trim();

  function t(s, vars) {
    let out = lang === "ru" && dict[s] !== undefined ? dict[s] : s;
    if (vars) out = out.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
    return out;
  }

  function apply() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      if (el.dataset.i18nEn === undefined) el.dataset.i18nEn = el.innerHTML;
      const key = el.dataset.i18n || norm(el.dataset.i18nEn);
      el.innerHTML = lang === "ru" && dict[key] !== undefined ? dict[key] : el.dataset.i18nEn;
    });
    if (toggle) {
      toggle.textContent = lang === "ru" ? "EN" : "RU";
      toggle.setAttribute("aria-label", lang === "ru" ? "Switch to English" : "Переключить на русский");
    }
    listeners.forEach((fn) => fn(lang));
  }

  function setLang(next) {
    lang = next;
    try { localStorage.setItem("lang", lang); } catch (e) { /* storage blocked */ }
    apply();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const header = document.querySelector(".site-header");
    if (header) {
      toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "lang-toggle";
      toggle.addEventListener("click", () => setLang(lang === "ru" ? "en" : "ru"));
      header.appendChild(toggle);
    }
    apply();
  });

  window.I18N = {
    t,
    add: (entries) => Object.assign(dict, entries),
    onChange: (fn) => listeners.push(fn),
    lang: () => lang,
  };
})();
