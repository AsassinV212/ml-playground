# Adding a topic

Each topic is made of three pieces that mirror each other:

1. **Page** — `topics/<topic>.html` with its script `topics/<topic>.js`.
   Link `../assets/style.css` and `../assets/common.js` for the shared look
   and canvas helpers. Keep everything dependency-free.
2. **Reference** — `python/<topic>.py`, numpy only, written to be read.
   Its formulas must match the JavaScript on the page.
3. **Tests** — `python/test_<topic>.py`, run with `pytest python`.

4. **Russian translation** — `topics/<topic>.ru.js`, see below.

Then:

- Turn the topic's "coming soon" card in `index.html` into a link.
- Add a row to the table in `README.md` and a line to `README.ru.md`.

## Translation (RU/EN toggle)

`assets/i18n.js` adds a RU/EN button to every `.site-header` and swaps text without a reload.

- Write English in the HTML. Mark translatable elements with `data-i18n`:
  - `<h1 data-i18n>Decision tree</h1>`: the English text itself is the key;
  - `<article class="prose" data-i18n="tree.article">`: an explicit key for long blocks.
  Do not nest `data-i18n` elements; wrap plain text next to an `<output>` or `<input>` in a `<span data-i18n>`.
- Put the Russian in `topics/<topic>.ru.js` with `I18N.add({ key: "русский текст" })`.
  Strings shared by several pages go in `assets/ru-common.js`.
- Text built in JavaScript goes through `ML.t("Converged after {n} steps.", { n })`.
  Use `ML.statusLine(el)` for status messages and `ML.onLang(render)` to redraw canvases on switch.
- Script order: `i18n.js`, `ru-common.js`, `<topic>.ru.js`, `common.js`, `<topic>.js`.

## Page checklist

- Works at 360 px width; no horizontal scroll.
- Controls are at least 44 px tall (comfortable for a finger).
- Canvas drawing uses `fitCanvas()` so it stays sharp on Retina screens.
- A short explanation section under the demo, plus a link to the Python reference.
- A `.source` box at the end linking the matching chapter of the
  [Yandex ML handbook](https://contest.yandex.ru/tracks/ml/). Explain in our own words;
  never copy text from the handbook.
- Asset links carry a version, e.g. `common.js?v=2`. GitHub Pages lets browsers cache
  files for 10 minutes, so after changing `assets/common.js` or `assets/style.css`
  bump the version in **every** page; otherwise phones keep the old helpers and the
  new pages break.
- Shared helpers in `assets/common.js` (`plot2d`, `drawGrid`, `drawPoint`, `hitPoint`,
  `segmented`, `rng`, ...) cover most 2D demos; reuse them before writing new ones.
