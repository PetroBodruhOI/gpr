import uuid, os
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import aiofiles, redis.asyncio as aioredis
from .schemas import PredictUrlRequest, TaskStatus
from .tasks import celery_app, run_predict

app = FastAPI(title="Guitar Pattern Classifier", version="1.0.0")

_allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)

# Prometheus — три рядки
Instrumentator().instrument(app).expose(app)

redis = aioredis.from_url(os.environ["REDIS_URL"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict/file")
async def predict_file(file: UploadFile):
    task_id = str(uuid.uuid4())
    # Зберігаємо файл у спільну папку
    uploads_dir = os.environ.get("UPLOADS_DIR", "/uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, f"{task_id}_{file.filename}")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await file.read())

    run_predict.delay(task_id, audio_path=file_path)
    return {"task_id": task_id}


@app.post("/predict/url")
async def predict_url(req: PredictUrlRequest):
    task_id = str(uuid.uuid4())
    run_predict.delay(task_id, url=req.url,
                      start_sec=req.start_sec,
                      duration_sec=req.duration_sec)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    raw = await redis.get(f"task:{task_id}")
    if raw is None:
        return TaskStatus(task_id=task_id, status="pending", progress=0)
    import json
    return TaskStatus(task_id=task_id, **json.loads(raw))
