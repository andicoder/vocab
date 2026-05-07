from fastapi.testclient import TestClient

from vocab_api.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vocab_requires_auth():
    response = client.get("/vocab")
    assert response.status_code == 401
    assert "x-authentik-username" in response.json()["detail"].lower()
