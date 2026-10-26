# ml-playground

Интерактивные визуализации алгоритмов машинного обучения без внешних зависимостей.
Каждая тема — одна страница, с которой можно поиграть прямо на телефоне,
и короткая понятная реализация на Python (только numpy).

*English version: [README.md](README.md)*

## Темы

Темы идут по [учебнику ML от ШАД Яндекса](https://contest.yandex.ru/tracks/ml/);
на каждой странице есть ссылка на соответствующую главу.

1. Градиентный спуск — [страница](topics/gradient-descent.html), [код](python/gradient_descent.py)
2. Линейная регрессия — [страница](topics/linear-regression.html), [код](python/linear_regression.py)
3. Логистическая регрессия — [страница](topics/logistic-regression.html), [код](python/logistic_regression.py)
4. Метод k ближайших соседей — [страница](topics/knn.html), [код](python/knn.py)
5. Решающее дерево — [страница](topics/decision-tree.html), [код](python/decision_tree.py)
6. Переобучение — [страница](topics/overfitting.html), [код](python/overfitting.py)

Скоро: ансамбли и градиентный бустинг, метрики классификации, маленькая нейросеть.

## Запуск

Сайт: откройте `index.html` в браузере или запустите `python -m http.server 8000`
и перейдите на <http://localhost:8000>.

Тесты Python:

```bash
python -m venv .venv
.venv/bin/pip install -r python/requirements.txt
.venv/bin/python -m pytest python
```

(На Windows вместо `.venv/bin/` пишите `.venv\Scripts\`.)

## Публикация на GitHub Pages

Settings → Pages → Deploy from a branch → `main`, папка `/ (root)`.
Сайт появится по адресу `https://<ваш-логин>.github.io/ml-playground/`.
