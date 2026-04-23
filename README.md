# Employment Dashboard

Веб-приложение для анализа трудоустройства выпускников: обучает ML-модели (Logistic Regression, LightGBM, CatBoost), показывает What-If прогноз, группы риска, когортные сравнения и интегрируется с Trudvsem.ru для подсчёта вакансий.

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Откройте http://localhost:8000 — при первом открытии демо-данные загрузятся автоматически.

## Деплой на Render.com

1. Запушьте репозиторий на GitHub.
2. На [render.com](https://render.com) → **New** → **Web Service** → подключите репозиторий.
3. Заполните форму:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. Нажмите **Create Web Service**. Билд занимает 5–10 минут (тяжёлые ML-зависимости).
5. После завершения получите ссылку вида `https://your-app.onrender.com`.

Free-инстанс засыпает через 15 мин неактивности. Первый запрос после паузы займёт ~30 сек, дальше работает мгновенно.

## Структура

- `app.py` — FastAPI бэкенд: обучение моделей, /api/predict, /api/risk_groups, /api/vacancies. При старте автоматически тренируется на `synthetic_10000.xlsx`.
- `static/dashboard.html` — единственный фронтенд-файл (HTML + CSS + JS, Chart.js, SheetJS).
- `static/synthetic_10000.xlsx` — демо-датасет на 10 000 записей.
- `requirements.txt` — зависимости.
- `runtime.txt` / `Procfile` — конфигурация для Render.

## Эндпоинты API

- `POST /api/preprocess` — нормализация загруженных строк
- `POST /api/train` — обучение моделей
- `POST /api/predict` — вероятность трудоустройства (What-If)
- `POST /api/risk_groups` — выпускники в группе риска (individual / group режимы)
- `POST /api/vacancies` — релевантные вакансии направлению через Trudvsem.ru
