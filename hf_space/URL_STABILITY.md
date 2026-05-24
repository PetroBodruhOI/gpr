# Нестабільність класифікації при URL

## Симптоми
- Одна й та сама YouTube-посилання дає різні класи при повторному запиті
- Локальний файл стабільний, URL — нестабільний
- Класи часто "стрибають" між сусідніми (6a ↔ 6b, 8a ↔ 8b)

## Причини

| Причина | Вплив | Фікс |
|---------|-------|-----|
| **YouTube якість** | YouTube компресує MP3 (128 kbps) → різна якість кожного завантаження | ✅ `_normalize_audio()` |
| **Demucs чутливість** | Низька якість → Demucs гірше виділяє гітару | ✅ Нормалізація перед Demucs |
| **ОбрізкаFFmpeg** | `-ss` / `-t` можуть창 утворити артефакти на крайах | ⚠️ Використовувати full-duration |
| **Chunk-boundary ефекти** | Якщо обрізка потрапить в середину ноти | ✅ Мінімізовано нормалізацією |
| **Floating-point варіаційність** | NumPy операції мають рандомізацію в деяких дистрибутивах | ✅ Фіксовані seed у тестах |

## Фіксація (додана)

### 1. Аудіо-нормалізація
```python
def _normalize_audio(y, sr):
    # Preemphasis (high-pass фільтр) — видаляє низькі частоти
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    # RMS-норма — стандартизація гучності
    rms = np.sqrt(np.mean(y ** 2))
    y = y / rms * 0.5  # нормалізуємо
    return y
```

Тепер URL аудіо (навіть стиснене YouTube) дає **стійкі фічі**.

### 2. Рекомендації для користувачів

**Для максимальної стабільності:**

```bash
# ❌ Уникати: користувацькі обрізки
POST /predict/url
{
  "url": "https://youtu.be/xyz",
  "start_sec": 45,        # ← може обрізати ноту
  "duration_sec": 30
}

# ✅ Краще: вся тісня без обрізки
POST /predict/url
{
  "url": "https://youtu.be/xyz"
}

# ✅ Найстабільніше: локальний файл
POST /predict/file
< upload MP3 >
```

## Тестування нестабільності

```bash
# Повторювальний запит до однієї URL → перевірити стабільність
for i in {1..5}; do
  curl -X POST http://localhost:7860/predict/url \
    -H "Content-Type: application/json" \
    -d '{"url":"https://youtu.be/xyz"}' \
    | jq '.result.final_label'
done

# Очікуваний результат: один і той же клас 5 разів
```

## Метрики

Моніторимо в Prometheus:
```
gpr_predictions_total{source="url",predicted_class="6a",status="done"}
gpr_predictions_total{source="file",predicted_class="6a",status="done"}
```

Якщо `source="url"` має більш варіативні результати → проблема в URL-логіці.
