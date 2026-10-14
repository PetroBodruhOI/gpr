import os

import pytest

# app.py reads REDIS_URL only in lifespan, but set a dummy so imports never fail.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


class MockRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = value

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self.store if k.startswith(prefix)]


@pytest.fixture
def mock_redis():
    return MockRedis()
