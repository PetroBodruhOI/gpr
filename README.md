# GPR — Guitar Pattern Recommender

![CI](https://github.com/yourname/guitar-classifier/actions/workflows/ci.yml/badge.svg)

Класифікатор / рекомендер гітарних патернів — дипломна робота.

## Архітектура

```
User
 │
 ▼
Vercel (React + Vite SPA)
 │  POST /predict/file
 │  GET  /tasks/{id}            (polling progress)
 │  POST /feedback/{id}         (👍 / 👎)
 ▼
HuggingFace Space — Docker, FastAPI
 │  HTDemucs(_6s) → BeatThis → 46 ознак на 6-с chunk → LightGBM (soft voting)
 │  завантажує model.pkl з HF Hub при старті
 └──▶ Upstash Redis  (task state + progress)
```

**Той самий код** працює локально (через Docker) і в проді (Vercel + HF Space). Без двох версій логіки.

## Структура репозиторію

| Шлях | Призначення |
|------|-------------|
| `frontend/` | React + Vite + TS — Vercel у проді, у локальному Docker через vite dev server |
| `backend/app.py` | FastAPI: endpoints, фонова задача, Prometheus-метрики |
| `backend/ml_pipeline/` | `preprocess.py` (Demucs, BeatThis) · `feature_extract.py` (46 ознак) · `classify.py` (chunks + soft voting) |
| `backend/tests/` | pytest для backend |
| `training/` | Офлайн-навчання (`train.py extract` / `train`). Не деплоїться; використовує `backend/ml_pipeline`, тож ознаки рахуються тим самим кодом |
| `docker-compose.yml` | Локальний стек (redis + backend + frontend) |
| `.github/workflows/ci.yml` | CI: тести + auto-deploy у HF Space і Vercel на push до `main` |

## Локальний запуск

```bash
docker compose up --build
```
- Frontend: http://localhost:5173
- Worker API: http://localhost:7860  (Swagger: `/docs`)
- Redis: localhost:6379

Перший запуск тягне demucs `htdemucs_6s` (~2 GB) і `model.pkl` з HF Hub → потім кешується у volumes.

## Деплой у production

| Сервіс | Платформа | Ціна |
|---|---|---|
| Frontend | Vercel (auto on git push to `main`) | $0 |
| Worker | HuggingFace Space (CPU Basic free, або T4 paid) | $0 / $0.60/год |
| Redis | Upstash (free tier) | $0 |
| Model | HuggingFace Model Hub | $0 |

Деплой автоматичний: push / merge у `main` → CI проганяє тести і пушить `backend/` у HF Space, а `frontend/` — у Vercel.

## Навчання моделі

Будь-який sample rate і кількість каналів (44.1k, 48k, стерео…) автоматично зводяться до 22 050 Гц моно — і в навчанні, і в проді.

```bash
cd training
pip install -r ../backend/requirements-dev.txt
python train.py extract --audio_dir ./audio --output features.csv --use_demucs --use_beats
python train.py train   --csv features.csv --output model.pkl
```
Отриманий `model.pkl` завантажується в HF Model Hub (`HF_MODEL_REPO`).

## Тести

```bash
cd backend && pip install -r requirements-dev.txt && pytest
cd training && pytest
cd frontend && npm ci && npm test
```

## Environment variables

### Production
| Місце | Змінна | Опис |
|---|---|---|
| Vercel | `VITE_API_URL` | URL HF Space, напр. `https://petrobodruh-gpr-worker.hf.space` |
| HF Space | `REDIS_URL` | Upstash URL (`rediss://default:...@xxx.upstash.io:6379`) |
| HF Space | `HF_MODEL_REPO` | `PetroBodruh/grp-model` |
| HF Space | `HF_TOKEN` | Тільки якщо model repo приватна |
| HF Space | `ALLOWED_ORIGINS` | `https://gpr.vercel.app,http://localhost:5173` |
| HF Space | `METRICS_USER` / `METRICS_PASSWORD` | Basic-auth для `/metrics`. Не задані → `/metrics` вимкнено (404) |
| HF Space | `MAX_UPLOAD_MB` | Ліміт розміру файлу (за замовч. 100) |

### Local Docker
Все вже у `docker-compose.yml`. Може треба додати `HF_TOKEN` якщо model repo приватна.
