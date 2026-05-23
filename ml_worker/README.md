# ML Worker (local Docker dev)

Celery-воркер, що виконує ML-пайплайн локально (через `docker-compose up`).

> ⚠ **Production живе у [`../hf_space/`](../hf_space/)** — FastAPI-додаток на HuggingFace Space.
> Ця тека потрібна тільки для локальної розробки через Docker.

## Архітектура

1. Backend (FastAPI у `../backend/`) ставить задачу в Redis-чергу.
2. Цей worker (Celery) забирає задачу, виконує:
   - `yt-dlp` — завантаження аудіо з URL (опційно);
   - HTDemucs (`htdemucs_6s`) — виділення гітарного стему;
   - BeatThis — детекція beat/downbeat;
   - LightGBM — класифікація по чанках + soft voting;
3. Прогрес та результат пишуться у Redis з ключем `task:{task_id}`.

## Environment variables

| Назва         | Опис |
|---------------|------|
| `REDIS_URL`   | Redis сервісу (з docker-compose або Upstash) |
| `MODEL_PATH`  | Шлях до `model.pkl` (default `./model.pkl`) |
| `HF_MODEL_REPO` | Опційно: тягнути модель з HF Hub якщо локального файлу нема |
| `HF_TOKEN`    | Опційно: токен для приватного repo |

## Локальний запуск (без docker-compose)

```bash
docker build -t gpr-worker .
docker run -e REDIS_URL=redis://host.docker.internal:6379 gpr-worker
```
