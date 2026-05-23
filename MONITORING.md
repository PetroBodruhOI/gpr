# Monitoring guide — GPR

## Що збирає бекенд

| Метрика | Що показує |
|---|---|
| `gpr_predictions_total` | К-сть запитів по класах, source, status (done/error) |
| `gpr_inference_duration_seconds` | Загальний час inference |
| `gpr_stage_duration_seconds` | Час кожного етапу (download / demucs / beat / classify) |
| `gpr_audio_duration_seconds` | Тривалість вхідного аудіо |
| `gpr_inference_confidence` | Розподіл confidence по класах |
| `gpr_user_feedback_total` | 👍/👎 від користувачів |

**Endpoint**: `https://petrobodruh-gpr-worker.hf.space/metrics`

Логи — JSON у stdout, видно у HF Space → **Logs** tab.

---

## 🟢 Найпростіший спосіб (для дипломної)

**Відкрий URL у браузері і покажи на захисті:**

```
https://petrobodruh-gpr-worker.hf.space/metrics
```

Це plain text з усіма лічильниками. Цього достатньо щоб у дипломі написати:
> *"Реалізовано експорт у Prometheus-форматі через endpoint `/metrics`. Будь-який Prometheus-сервер може скрейпити ці дані."*

Все. Дальше — опційно.

---

## 🟡 Якщо хочеш бачити метрики у GitHub

Так, **це можливо**. GitHub Action раз на годину тягне `/metrics`, парсить ключові цифри і публікує summary у Actions tab.

Створи `.github/workflows/metrics-snapshot.yml`:

```yaml
name: Metrics snapshot

on:
  schedule:
    - cron: '0 * * * *'   # щогодини
  workflow_dispatch:

jobs:
  snapshot:
    runs-on: ubuntu-latest
    steps:
      - name: Fetch metrics
        run: |
          curl -s https://petrobodruh-gpr-worker.hf.space/metrics > metrics.txt
          echo "## GPR metrics snapshot at $(date -u +%FT%TZ)" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
          grep -E '^gpr_' metrics.txt >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
      - uses: actions/upload-artifact@v4
        with:
          name: metrics-${{ github.run_id }}
          path: metrics.txt
```

Що це робить:
- щогодини запит до `/metrics`
- лічильники `gpr_*` показуються у GitHub Actions → конкретний run → **Summary** (видно як таблиця)
- raw текст зберігається у Artifacts на 90 днів

**Не вимагає Grafana, не вимагає Docker, не вимагає VPS.** Все живе у GitHub.

Для дипломної цього **досить багато**. На захисті відкриваєш Actions tab і показуєш як метрики накопичуються.

---

## 🔵 Якщо хочеш реальні дашборди — Grafana Cloud

UI у Grafana Cloud сильно змінився, тому даю принципи замість конкретних menu paths.

### 1. Реєстрація

1. https://grafana.com/auth/sign-up/create-user
2. Підтверди email
3. Створи **stack** (organization) — Grafana автоматично сама зробить це при першому вході
4. У портал-екрані з'явиться твій stack (наприклад `gpr-monitoring.grafana.net`)

### 2. Знайди credentials

Поточний UI ховає credentials у двох місцях:

- **Prometheus URL** і **username**: Stack details (іконка `⋯` біля назви stack'у) → "Details" або "View details"
- **Access token (password для basic_auth)**:
  - Альо новий шлях: **Portal → Security → Access Policies** → створи нову policy зі scope **`metrics:write`** і **`logs:write`** → потім **Add token** → копіюй
  - Або старіший шлях: **Portal → API Keys** (deprecated але ще працює)

Запам'ятай:
```
PROMETHEUS_URL = https://prometheus-prod-XX-prod-eu-west-X.grafana.net/api/prom/push
PROMETHEUS_USER = 1234567       (число)
PROMETHEUS_TOKEN = glc_xxx...   (string з префіксом glc_)
```

### 3. Запусти Grafana Agent щоб скрейпив твій HF Space

Grafana Agent — це маленький контейнер який раз на 30с тягне `/metrics` і пушить у Grafana Cloud. Запускати треба десь де є інтернет (твій комп, VPS, навіть GitHub Action).

#### Варіант 3A — локально через Docker (на твоєму комп'ютері)

`monitoring/agent.yaml`:

```yaml
metrics:
  configs:
    - name: gpr
      scrape_configs:
        - job_name: gpr-worker
          scrape_interval: 30s
          static_configs:
            - targets: ['petrobodruh-gpr-worker.hf.space:443']
          scheme: https
          metrics_path: /metrics
      remote_write:
        - url: PROMETHEUS_URL_TUT
          basic_auth:
            username: 'PROMETHEUS_USER_TUT'
            password: 'PROMETHEUS_TOKEN_TUT'
```

Запуск:
```bash
docker run -d --name grafana-agent \
  -v ${PWD}/monitoring/agent.yaml:/etc/agent/agent.yaml \
  grafana/agent:latest \
  -config.file=/etc/agent/agent.yaml
```

⚠ Працює тільки коли твій ПК увімкнений з інтернетом. Для презентації — увімкни заздалегідь.

#### Варіант 3B — GitHub Actions як scraper

Більш складно, бо треба самим зробити Prometheus remote_write (protobuf+snappy). Я б не рекомендував — швидше підняти Agent у Docker.

### 4. Створення дашбордів

У Grafana (`yourstack.grafana.net`) → **Dashboards** → **+ New dashboard** → **Add visualization** → select Prometheus data source (вже доданий автоматично).

Запити (Query):

```promql
# Скільки запитів обробив за добу, по класах
sum(increase(gpr_predictions_total{status="done"}[24h])) by (predicted_class)

# p50 / p95 час inference
histogram_quantile(0.95, sum(rate(gpr_inference_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.50, sum(rate(gpr_inference_duration_seconds_bucket[5m])) by (le))

# Розподіл confidence по класах
sum(rate(gpr_inference_confidence_bucket[1h])) by (predicted_class, le)

# % позитивного фідбеку
sum(rate(gpr_user_feedback_total{rating="good"}[24h])) /
sum(rate(gpr_user_feedback_total[24h])) * 100

# Скільки часу займає кожен етап (медіана)
histogram_quantile(0.50, sum(rate(gpr_stage_duration_seconds_bucket[5m])) by (stage, le))
```

---

## Що показати на захисті

Залежно скільки вкладеш часу:

| Час вкладено | Що покажеш |
|---|---|
| **0 хв** (нічого не робиш окрім коду який вже є) | URL `/metrics` у браузері: "є експорт у Prometheus" |
| **15 хв** (додаси GitHub Action snapshot) | Actions tab з історією метрик щогодини |
| **1-2 год** (Grafana Cloud + Agent) | Реальні дашборди з графіками |

Для дипломної **15-хвилинного варіанту досить**. Не витрачай вечір на Grafana якщо є інші завдання.

---

## Який варіант обрати

- **Просто хочу галочку у дипломі** → варіант 🟢 (URL у браузері)
- **Хочу історію метрик без зайвих сервісів** → варіант 🟡 (GitHub Action)
- **Хочу реальні графіки для wow-ефекту на захисті** → варіант 🔵 (Grafana Cloud)
