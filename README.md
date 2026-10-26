# ml-playground

Interactive, dependency-free visualizations of machine learning algorithms.
Made for students: every topic is a single page you can poke at on your phone,
plus a short, readable Python reference implementation.

*Русская версия: [README.ru.md](README.ru.md)*

## Topics

| Topic | Page | Python reference |
|---|---|---|
| Gradient descent | [topics/gradient-descent.html](topics/gradient-descent.html) | [python/gradient_descent.py](python/gradient_descent.py) |
| Linear regression | coming soon | |
| Logistic regression | coming soon | |
| Overfitting | coming soon | |
| k-nearest neighbors | coming soon | |
| Decision tree | coming soon | |

## Run the site locally

The site is plain HTML, CSS and JavaScript: no build step, no CDN, no packages.
Open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Run the Python reference and tests

The reference implementations use only `numpy`; the tests use `pytest`.

```bash
python -m venv .venv
.venv/bin/pip install -r python/requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest python                  # Windows: .venv\Scripts\python
```

## Deploy to GitHub Pages

Because the site is static, GitHub Pages can serve the repository as is:

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**,
   branch `main`, folder `/ (root)`, and save.
4. After a minute the site is live at `https://<your-user>.github.io/ml-playground/`.

All links in the site are relative, so it works from that sub-path without changes.

## Project structure

```
index.html          list of topics
topics/             one HTML page (+ its script) per topic
assets/             shared CSS and JavaScript helpers
python/             numpy reference implementations and pytest tests
docs/               notes for contributors
```

See [docs/adding-a-topic.md](docs/adding-a-topic.md) to add a new topic.
