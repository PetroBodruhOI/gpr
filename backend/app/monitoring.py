from prometheus_client import Counter, Histogram

# Кастомні метрики поверх дефолтних від Instrumentator
predict_requests_total = Counter(
    "predict_requests_total",
    "Кількість запитів на /predict",
    ["endpoint", "status"],
)

predict_latency_seconds = Histogram(
    "predict_latency_seconds",
    "Час обробки запиту /predict (від API до постановки в чергу)",
    ["endpoint"],
)

task_completion_total = Counter(
    "task_completion_total",
    "Кількість завершених задач",
    ["status"],  # done | error
)

task_duration_seconds = Histogram(
    "task_duration_seconds",
    "Тривалість виконання задачі у worker (Demucs + BeatThis + LightGBM)",
    buckets=[1, 2, 5, 10, 30, 60, 120, 300],
)
