import os
from celery import Celery

REDIS_URL = os.environ["REDIS_URL"]

celery_app = Celery(
    "guitar_classifier",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Сигнатура задачі (реальна реалізація — у ml_worker/worker.py).
# Бекенд лише ставить задачу в чергу через .delay(), worker її виконує.
@celery_app.task(name="run_predict")
def run_predict(task_id, audio_path=None, url=None,
                start_sec=None, duration_sec=None):
    raise NotImplementedError(
        "run_predict виконується у ml_worker, не у backend"
    )
