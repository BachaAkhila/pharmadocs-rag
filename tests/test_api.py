from fastapi.testclient import TestClient

from app.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pipeline_ready"] is True


def test_query_endpoint_returns_answer_with_chunks():
    with TestClient(app) as client:
        response = client.post("/query", json={"query": "What MAC reduction was achieved on ResNet-56?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["confidence"] >= 0
    assert len(body["retrieved_chunks"]) > 0


def test_agent_query_endpoint_returns_attempts():
    with TestClient(app) as client:
        response = client.post("/agent/query", json={"query": "What MAC reduction was achieved on ResNet-56?"})

    assert response.status_code == 200
    body = response.json()
    assert body["attempts"] >= 1
    assert "low_confidence" in body


def test_query_endpoint_rejects_empty_query():
    with TestClient(app) as client:
        response = client.post("/query", json={"query": ""})

    assert response.status_code == 422


def test_metrics_endpoint_returns_baseline_if_present():
    with TestClient(app) as client:
        response = client.get("/metrics/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "no_baseline")
