# ml-playground

Интерактивные визуализации алгоритмов машинного обучения без внешних зависимостей.
Каждая тема — одна страница, с которой можно поиграть прямо на телефоне,
и короткая понятная реализация на Python (только numpy).

*English version: [README.md](README.md)*

## Темы

- Градиентный спуск — [topics/gradient-descent.html](topics/gradient-descent.html), код: [python/gradient_descent.py](python/gradient_descent.py)
- Скоро: линейная регрессия, логистическая регрессия, переобучение, kNN, дерево решений.

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
