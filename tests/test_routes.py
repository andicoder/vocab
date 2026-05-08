import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from vocab_api.audio import LocalDirAudioStorage, audio_key
from vocab_api.deps import get_gemini, get_storage, get_tts
from vocab_api.gemini import GeminiClient
from vocab_api.main import app


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def _gemini_with_handler(handler) -> tuple[GeminiClient, httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return (
        GeminiClient(
            http=http,
            api_key="k",
            model="m",
            base_url="https://example.com/v1beta",
        ),
        http,
    )


class _FakeTts:
    async def synthesize(self, *, text: str, voice: str) -> bytes:
        return b"FAKE-MP3:" + text.encode()


@pytest.fixture
def http_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_translate_route_requires_auth(http_client: TestClient):
    response = http_client.post("/translate", json={"word": "expedition"})
    assert response.status_code == 401


def test_translate_route_returns_translation(http_client: TestClient):
    payload = {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "die Reise",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(json.dumps(payload)))

    gemini, _ = _gemini_with_handler(handler)
    app.dependency_overrides[get_gemini] = lambda: gemini

    response = http_client.post(
        "/translate",
        json={"word": "expedition", "sentence": "A grand expedition north."},
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == payload


def test_audio_route_requires_auth(http_client: TestClient):
    response = http_client.get("/audio/expedition.mp3")
    assert response.status_code == 401


def test_audio_route_local_returns_file(http_client: TestClient, tmp_path: Path):
    storage = LocalDirAudioStorage(root=tmp_path, public_url_base="/audio")
    tts = _FakeTts()
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_tts] = lambda: tts

    response = http_client.get(
        "/audio/expedition.mp3",
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"FAKE-MP3:expedition"

    expected_key = audio_key("expedition", "en-US-AriaNeural", "en")
    assert (tmp_path / expected_key).exists()
