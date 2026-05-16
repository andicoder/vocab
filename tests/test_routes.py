import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from anki.collection import Collection
from fastapi.testclient import TestClient
from sqlalchemy import text

from vocab_api.anki_writer import AnkiWriter
from vocab_api.audio import AudioRequest, LocalDirAudioStorage, audio_key
from vocab_api.db import engine
from vocab_api.deps import get_anki_writer, get_gemini, get_storage, get_tts
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


_EMPTY_TRANSLATION_JSON = json.dumps(
    {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "",
        "ipa": "",
        "sense_key": "default",
        "sense_label": "",
        "collocations": [],
        "extra_examples": [],
    }
)


def stub_translation_gemini() -> GeminiClient:
    """Gemini stub for tests that exercise the approve path but don't care
    about translation content. Returns an empty translation on every call so
    `apply_approve`'s worker-field backfill (#58) doesn't hit the real API
    on a cache miss."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(_EMPTY_TRANSLATION_JSON))

    gemini, _ = _gemini_with_handler(handler)
    return gemini


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


def test_settings_get_returns_default_card_direction(http_client: TestClient):
    response = http_client.get("/me/settings", headers={"X-authentik-username": "alice"})
    assert response.status_code == 200, response.text
    assert response.json() == {"card_direction": "de_en"}


def test_settings_patch_updates_card_direction(http_client: TestClient):
    response = http_client.patch(
        "/me/settings",
        json={"card_direction": "both"},
        headers={"X-authentik-username": "alice"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"card_direction": "both"}
    # Persisted: a follow-up GET reflects the new value.
    response = http_client.get("/me/settings", headers={"X-authentik-username": "alice"})
    assert response.json() == {"card_direction": "both"}


def test_settings_patch_rejects_unknown_direction(http_client: TestClient):
    response = http_client.patch(
        "/me/settings",
        json={"card_direction": "es_de"},
        headers={"X-authentik-username": "alice"},
    )
    assert response.status_code == 422


def test_translate_route_returns_translation(http_client: TestClient):
    payload = {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "die Reise",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sense_key": "noun-journey",
        "sense_label": "Reise",
        "collocations": ["go on an expedition", "Arctic expedition"],
        "extra_examples": ["The Arctic expedition lasted three months."],
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

    expected_key = audio_key(AudioRequest(word="expedition"))
    assert (tmp_path / expected_key).exists()


class _RemoteFakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fetched: list[str] = []

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    async def fetch(self, key: str) -> bytes:
        self.fetched.append(key)
        return self.objects[key]

    def public_url(self, key: str) -> str:
        return f"https://private.example.com/{key}"


def test_audio_route_remote_streams_bytes_instead_of_redirecting(http_client: TestClient):
    storage = _RemoteFakeStorage()
    tts = _FakeTts()
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_tts] = lambda: tts

    response = http_client.get(
        "/audio/expedition.mp3",
        headers={"X-authentik-username": "alice"},
        follow_redirects=False,
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"FAKE-MP3:expedition"
    expected_key = audio_key(AudioRequest(word="expedition"))
    assert storage.fetched == [expected_key]


def test_approve_writes_anki_card_and_marks_synced(http_client: TestClient, tmp_path: Path):
    audio_dir = tmp_path / "audio"
    anki_root = tmp_path / "anki"
    storage = LocalDirAudioStorage(root=audio_dir, public_url_base="/audio")
    anki_writer = AnkiWriter(root=anki_root)
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer
    app.dependency_overrides[get_gemini] = stub_translation_gemini

    create = http_client.post(
        "/vocab",
        json={"word": "expedition", "sentence": "A grand expedition north."},
        headers={"X-authentik-username": "alice"},
    )
    assert create.status_code == 201, create.text
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={
            "lemma": "expedition",
            "translation": "die Expedition",
            "alternatives": "die Reise",
            "ipa": "/ˌɛkspɪˈdɪʃən/",
        },
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "synced"
    assert body["anki_card_id"] is not None
    assert body["approved_at"] is not None
    assert body["synced_at"] is not None
    assert (anki_root / "alice" / "collection.anki2").exists()


def test_approve_rejects_untranslated_entry(http_client: TestClient, tmp_path: Path):
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer

    create = http_client.post(
        "/vocab",
        json={"word": "expedition"},
        headers={"X-authentik-username": "alice"},
    )
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={},
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 400


def test_approve_backfills_cloze_when_entry_missing_it(http_client: TestClient, tmp_path: Path):
    # Regression: entries that landed in `needs-review` before #23 lack a
    # cloze_sentence. Approving them today must derive one (via the regex
    # mask on the source sentence) before the note hits Anki — otherwise the
    # DE→EN front renders an empty `<p>`.
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer
    app.dependency_overrides[get_gemini] = stub_translation_gemini

    create = http_client.post(
        "/vocab",
        json={"word": "expedition", "sentence": "A grand expedition north."},
        headers={"X-authentik-username": "alice"},
    )
    assert create.status_code == 201
    entry_id = create.json()["id"]

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE vocab.entry SET status='needs-review', cloze_sentence=NULL"
                    f" WHERE id = {entry_id}"
                )
            )

    asyncio.run(_seed())

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={"lemma": "expedition", "translation": "die Expedition"},
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    card_id = response.json()["anki_card_id"]
    col = Collection(str(tmp_path / "anki" / "alice" / "collection.anki2"))
    try:
        note = col.get_card(card_id).note()
        assert note["ClozeSentence"] == "A grand ___ north."
    finally:
        col.close()


def test_approve_backfills_extra_fields_from_translation_cache(
    http_client: TestClient, tmp_path: Path
):
    # Regression for #58: pre-#23 needs-review entries lack extra_examples,
    # collocations and sense_label — the worker writes them, but legacy rows
    # never went through it. Approving such an entry today must backfill
    # those fields from translate_with_cache before the Anki note is written.
    # The cache is seeded directly so the path doesn't need a Gemini call.
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer

    create = http_client.post(
        "/vocab",
        json={"word": "expedition", "sentence": "A grand expedition north."},
        headers={"X-authentik-username": "alice"},
    )
    assert create.status_code == 201
    entry_id = create.json()["id"]

    from vocab_api.gemini import _sentence_hash  # noqa: PLC0415 — local helper, test-only use

    sh = _sentence_hash("A grand expedition north.")
    assert sh is not None

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE vocab.entry SET status='needs-review',"
                    " extra_examples=NULL, collocations=NULL, sense_label=NULL,"
                    " cloze_sentence='A grand ___ north.'"
                    f" WHERE id = {entry_id}"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.translation_cache"
                    " (word, sentence_hash, lang, lemma, translation,"
                    "  alternatives, ipa, sense_key, sense_label,"
                    "  collocations, extra_examples)"
                    " VALUES ('expedition', :sh, 'en', 'expedition',"
                    " 'die Expedition', '', '', 'noun-journey', 'Reise',"
                    " 'go on an expedition · Arctic expedition',"
                    " 'The Arctic expedition lasted three months.')"
                ),
                {"sh": sh},
            )

    asyncio.run(_seed())

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={"lemma": "expedition", "translation": "die Expedition"},
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    card_id = response.json()["anki_card_id"]
    col = Collection(str(tmp_path / "anki" / "alice" / "collection.anki2"))
    try:
        note = col.get_card(card_id).note()
        assert note["ExtraExamples"] == "The Arctic expedition lasted three months."
        assert note["Collocations"] == "go on an expedition · Arctic expedition"
        assert note["SenseLabel"] == "Reise"
    finally:
        col.close()


def test_reject_marks_rejected(http_client: TestClient):
    create = http_client.post(
        "/vocab",
        json={"word": "expedition"},
        headers={"X-authentik-username": "alice"},
    )
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/vocab/{entry_id}/reject",
        headers={"X-authentik-username": "alice"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "rejected"


def test_approve_other_users_entry_404(http_client: TestClient, tmp_path: Path):
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer

    create = http_client.post(
        "/vocab",
        json={"word": "expedition"},
        headers={"X-authentik-username": "alice"},
    )
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={"lemma": "x", "translation": "y"},
        headers={"X-authentik-username": "bob"},
    )
    assert response.status_code == 404
