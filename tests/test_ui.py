import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from vocab_api.anki_writer import AnkiWriter
from vocab_api.audio import LocalDirAudioStorage
from vocab_api.db import engine
from vocab_api.deps import get_anki_writer, get_storage
from vocab_api.main import app


@pytest.fixture
def http_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _alice() -> dict:
    return {"X-authentik-username": "alice"}


def test_index_page_requires_auth(http_client: TestClient):
    response = http_client.get("/")
    assert response.status_code == 401


def test_index_page_renders_form_in_default_locale(http_client: TestClient):
    response = http_client.get("/", headers=_alice())
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'lang="de"' in response.text
    assert "Neues Wort" in response.text
    assert 'name="word"' in response.text
    assert 'hx-post="/ui/vocab"' in response.text


def test_index_page_honours_accept_language(http_client: TestClient):
    response = http_client.get("/", headers={**_alice(), "Accept-Language": "en-US,en;q=0.9"})
    assert response.status_code == 200
    assert 'lang="en"' in response.text
    assert "Add a word" in response.text


def test_queue_page_empty_state(http_client: TestClient):
    response = http_client.get("/queue", headers=_alice())
    assert response.status_code == 200
    assert "Keine Einträge zur Review." in response.text


def test_htmx_create_entry_returns_toast(http_client: TestClient):
    response = http_client.post(
        "/ui/vocab",
        data={"word": "expedition", "sentence": "A grand expedition.", "source": "test"},
        headers=_alice(),
    )
    assert response.status_code == 200
    assert "expedition" in response.text
    assert "hinzugefügt" in response.text


def test_queue_page_lists_needs_review_entries(http_client: TestClient):
    create = http_client.post("/vocab", json={"word": "expedition"}, headers=_alice())
    assert create.status_code == 201
    entry_id = create.json()["id"]

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE vocab.entry SET status='needs-review',"
                    " lemma='expedition', translation='die Expedition'"
                    f" WHERE id = {entry_id}"
                )
            )

    asyncio.run(_seed())

    response = http_client.get("/queue", headers=_alice())
    assert response.status_code == 200
    assert "expedition" in response.text
    assert "Übernehmen" in response.text
    assert "Verwerfen" in response.text


def test_htmx_approve_writes_anki_and_clears_row(http_client: TestClient, tmp_path: Path):
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer

    create = http_client.post("/vocab", json={"word": "expedition"}, headers=_alice())
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/ui/vocab/{entry_id}/approve",
        data={
            "lemma": "expedition",
            "translation": "die Expedition",
            "alternatives": "die Reise",
            "ipa": "/ˌɛkspɪˈdɪʃən/",
        },
        headers=_alice(),
    )
    assert response.status_code == 200, response.text
    assert response.text == ""
    assert (tmp_path / "anki" / "alice" / "collection.anki2").exists()


def test_htmx_reject_clears_row(http_client: TestClient):
    create = http_client.post("/vocab", json={"word": "x"}, headers=_alice())
    entry_id = create.json()["id"]

    response = http_client.post(f"/ui/vocab/{entry_id}/reject", headers=_alice())
    assert response.status_code == 200
    assert response.text == ""

    listing = http_client.get("/vocab?status=rejected", headers=_alice())
    assert listing.status_code == 200
    assert any(e["id"] == entry_id for e in listing.json())
