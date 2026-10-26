# ml-playground

Interactive, dependency-free visualizations of machine learning algorithms.
Made for students: every topic is a single page you can poke at on your phone,
plus a short, readable Python reference implementation.

*Русская версия: [README.ru.md](README.ru.md)*

## Topics

Topics follow the [Yandex ML handbook](https://contest.yandex.ru/tracks/ml/) (in Russian);
each page links to the matching chapter for the full theory. The explanations on the
pages are our own.

| # | Topic | Page | Python reference | Handbook |
|---|---|---|---|---|
| 1 | Gradient descent | [page](topics/gradient-descent.html) | [gradient_descent.py](python/gradient_descent.py) | §2.2, ch. 14 |
| 2 | Linear regression | [page](topics/linear-regression.html) | [linear_regression.py](python/linear_regression.py) | §2.2 |
| 3 | Logistic regression | [page](topics/logistic-regression.html) | [logistic_regression.py](python/logistic_regression.py) | §2.2 |
| 4 | k-nearest neighbors | [page](topics/knn.html) | [knn.py](python/knn.py) | §2.3 |
| 5 | Decision tree | [page](topics/decision-tree.html) | [decision_tree.py](python/decision_tree.py) | §2.4 |
| 6 | Overfitting | [page](topics/overfitting.html) | [overfitting.py](python/overfitting.py) | §3.3, §12.1 |

Coming soon: ensembles and gradient boosting, classification metrics, a tiny neural network.

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
