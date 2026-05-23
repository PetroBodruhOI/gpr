---
title: GPR Worker
emoji: 🎸
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# GPR Worker

FastAPI service that runs the Guitar Pattern Recommender pipeline:
HTDemucs → BeatThis → LightGBM classifier.

## Environment variables (set in Space settings)

| Name | Required | Example |
|------|----------|---------|
| `REDIS_URL` | yes | `rediss://default:xxx@xxx.upstash.io:6379` |
| `HF_MODEL_REPO` | yes | `your-username/gpr-model` |
| `HF_MODEL_FILE` | no  | `model.pkl` (default) |
| `HF_TOKEN` | only if model repo is private | `hf_xxxxx` |
| `ALLOWED_ORIGINS` | yes | `https://gpr.vercel.app,http://localhost:5173` |

## Endpoints

- `POST /predict/url`  — `{ url, start_sec?, duration_sec? }` → `{ task_id }`
- `POST /predict/file` — multipart form `file` → `{ task_id }`
- `GET  /tasks/{id}`   — polling endpoint with progress/result
- `GET  /`             — health check
