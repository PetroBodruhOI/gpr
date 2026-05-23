FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache/torch \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Pre-download Demucs models (htdemucs_6s has the dedicated guitar stem)
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

COPY . /app/

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--pool=solo"]
