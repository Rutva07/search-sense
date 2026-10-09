import os
from pathlib import Path
import uuid
import pytest
import redis
from fastapi.testclient import TestClient
from searchsense.history import RedisHistory, MemoryHistory
from searchsense.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store():
    url = os.environ.get("SEARCHSENSE_TEST_REDIS_URL", "redis://localhost:6379/15")
    namespace = "test-" + uuid.uuid4().hex
    try:
        history = RedisHistory(url, namespace, limit=3)
    except redis.RedisError:
        pytest.skip("Real Redis unavailable; set SEARCHSENSE_TEST_REDIS_URL")
    yield history
    keys = list(history.client.scan_iter(namespace + ":*"))
    if keys:
        history.client.delete(*keys)
    history.client.close()


def test_real_redis_history_asof_and_cache_version(store):
    memory = MemoryHistory(limit=3)
    for ts in range(10, 30):
        memory.add("user", ts, "hotel query " + str(ts))
        store.add("user", ts, "hotel query " + str(ts))
    assert [tuple(row) for row in store.get("user", 28)] == memory.get("user", 28)
    assert store.get("user", 28)[-1][0] == 27
    v = store.version("user")
    store.add("user", 31, "new observation")
    assert store.version("user") != v


def test_actual_http_api_cache_invalidation_and_validation(store):
    url = os.environ.get("SEARCHSENSE_TEST_REDIS_URL", "redis://localhost:6379/15")
    app = create_app(ROOT / "artifacts", url, store.namespace)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        payload = {"user_id": "new-user", "prefix": "best hotels in", "timestamp": 1148000000}
        first = client.post("/complete", json=payload)
        assert first.status_code == 200
        assert not first.json()["cache_hit"]
        assert first.json()["history_size"] == 0
        assert client.post("/complete", json=payload).json()["cache_hit"]
        response = client.post("/observe", json={"user_id": "new-user", "query": "best hotels in paris", "timestamp": 1147999990})
        assert response.status_code == 200
        next_result = client.post("/complete", json=payload).json()
        assert not next_result["cache_hit"]
        assert next_result["history_size"] == 1
        assert client.post("/complete", json={**payload, "prefix": "!!!!"}).status_code == 422
        assert client.post("/complete", json={**payload, "k": 0}).status_code == 422
        assert client.post("/observe", json={"user_id": "x", "query": "!!"}).status_code == 422
