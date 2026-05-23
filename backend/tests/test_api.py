"""Smoke-тести FastAPI ендпоінтів."""

import os
import pytest
from unittest.mock import patch, AsyncMock

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("redis.asyncio.from_url") as m:
        m.return_value = AsyncMock()
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_exists(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "python_info" in r.text or "process_" in r.text


def test_predict_url_returns_task_id(client):
    with patch("app.tasks.run_predict.delay") as m_delay:
        r = client.post("/predict/url", json={"url": "https://example.com/song"})
        assert r.status_code == 200
        body = r.json()
        assert "task_id" in body
        m_delay.assert_called_once()
