# Adding a topic

Each topic is made of three pieces that mirror each other:

1. **Page** — `topics/<topic>.html` with its script `topics/<topic>.js`.
   Link `../assets/style.css` and `../assets/common.js` for the shared look
   and canvas helpers. Keep everything dependency-free.
2. **Reference** — `python/<topic>.py`, numpy only, written to be read.
   Its formulas must match the JavaScript on the page.
3. **Tests** — `python/test_<topic>.py`, run with `pytest python`.

Then:

- Turn the topic's "coming soon" card in `index.html` into a link.
- Add a row to the table in `README.md` and a line to `README.ru.md`.

## Page checklist

- Works at 360 px width; no horizontal scroll.
- Controls are at least 44 px tall (comfortable for a finger).
- Canvas drawing uses `fitCanvas()` so it stays sharp on Retina screens.
- A short explanation section under the demo, plus a link to the Python reference.
- A `.source` box at the end linking the matching chapter of the
  [Yandex ML handbook](https://contest.yandex.ru/tracks/ml/). Explain in our own words;
  never copy text from the handbook.
- Shared helpers in `assets/common.js` (`plot2d`, `drawGrid`, `drawPoint`, `hitPoint`,
  `segmented`, `rng`, ...) cover most 2D demos; reuse them before writing new ones.
