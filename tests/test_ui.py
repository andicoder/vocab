import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.test_routes import stub_translation_gemini
from vocab_api.anki_writer import AnkiWriter
from vocab_api.audio import LocalDirAudioStorage
from vocab_api.db import engine
from vocab_api.deps import get_anki_writer, get_gemini, get_storage
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


def test_htmx_create_entry_renders_exists_toast_when_lemma_already_present(
    http_client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    # Regression for #10: when process_entry deletes the new row as a
    # duplicate of an existing lemma, the UI must render the "already exists"
    # toast instead of session.refresh()ing a deleted entry.
    from vocab_api.config import settings
    from vocab_api.routes import ui

    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    async def fake_process_entry(*, session, entry, user, deps):
        await session.delete(entry)
        return "dozen"

    monkeypatch.setattr(ui, "process_entry", fake_process_entry)

    response = http_client.post(
        "/ui/vocab",
        data={"word": "dozens", "sentence": "Dozens of pebbles."},
        headers=_alice(),
    )
    assert response.status_code == 200
    assert "dozens" in response.text
    assert "dozen" in response.text
    assert "schon" in response.text  # de: "schon als ... in der Sammlung"


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
    app.dependency_overrides[get_gemini] = stub_translation_gemini

    create = http_client.post(
        "/vocab",
        json={"word": "expedition", "sentence": "A grand expedition north."},
        headers=_alice(),
    )
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


def test_htmx_approve_returns_error_toast_when_not_translated(http_client: TestClient):
    # Regression for #52: approve must surface a visible toast when the entry
    # still lacks lemma/translation, not silently 400 with htmx ignoring the body.
    create = http_client.post("/vocab", json={"word": "expedition"}, headers=_alice())
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/ui/vocab/{entry_id}/approve",
        data={"lemma": "", "translation": ""},
        headers=_alice(),
    )
    assert response.status_code == 200, response.text
    assert response.headers["HX-Retarget"] == "#toast"
    assert response.headers["HX-Reswap"] == "innerHTML"
    assert "toast error" in response.text
    assert "noch nicht übersetzt" in response.text


def test_htmx_approve_returns_error_toast_when_anki_writer_raises(
    http_client: TestClient, tmp_path: Path
):
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")

    class BoomWriter:
        async def write_card(self, **_kwargs):
            raise RuntimeError("anki-sync unreachable")

    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: BoomWriter()

    create = http_client.post("/vocab", json={"word": "expedition"}, headers=_alice())
    entry_id = create.json()["id"]

    response = http_client.post(
        f"/ui/vocab/{entry_id}/approve",
        data={"lemma": "expedition", "translation": "die Expedition"},
        headers=_alice(),
    )
    assert response.status_code == 200, response.text
    assert response.headers["HX-Retarget"] == "#toast"
    assert "toast error" in response.text


def test_htmx_reject_returns_error_toast_when_entry_missing(http_client: TestClient):
    response = http_client.post("/ui/vocab/999999/reject", headers=_alice())
    assert response.status_code == 200, response.text
    assert response.headers["HX-Retarget"] == "#toast"
    assert response.headers["HX-Reswap"] == "innerHTML"
    assert "toast error" in response.text


def test_queue_page_has_toast_container(http_client: TestClient):
    response = http_client.get("/queue", headers=_alice())
    assert response.status_code == 200
    assert 'id="toast"' in response.text


def test_base_template_installs_htmx_response_error_listener(http_client: TestClient):
    response = http_client.get("/", headers=_alice())
    assert response.status_code == 200
    assert "htmx:responseError" in response.text


def test_index_page_prefills_query_params(http_client: TestClient):
    response = http_client.get(
        "/?word=expedition&sentence=A+grand+expedition&source=https%3A%2F%2Fexample.com",
        headers=_alice(),
    )
    assert response.status_code == 200
    assert 'value="expedition"' in response.text
    assert 'value="A grand expedition"' in response.text
    assert 'value="https://example.com"' in response.text


def test_static_files_served(http_client: TestClient):
    manifest = http_client.get("/static/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["start_url"] == "/"
    assert manifest.json()["display"] == "standalone"

    sw = http_client.get("/static/sw.js")
    assert sw.status_code == 200
    assert "serviceWorker" not in sw.text  # SW source itself, not registration
    assert "caches" in sw.text

    icon = http_client.get("/static/icon.svg")
    assert icon.status_code == 200
    assert "image/svg" in icon.headers["content-type"]


def test_base_template_links_manifest_and_registers_sw(http_client: TestClient):
    response = http_client.get("/", headers=_alice())
    assert response.status_code == 200
    assert '<link rel="manifest" href="/static/manifest.webmanifest">' in response.text
    assert 'navigator.serviceWorker.register("/static/sw.js")' in response.text
    assert '<meta name="theme-color"' in response.text


def test_bookmarklet_page_requires_auth(http_client: TestClient):
    assert http_client.get("/bookmarklet").status_code == 401


def test_bookmarklet_page_renders_javascript_link(http_client: TestClient):
    response = http_client.get(
        "/bookmarklet",
        headers={**_alice(), "Host": "vocab.example.com"},
    )
    assert response.status_code == 200
    assert 'href="javascript:' in response.text
    assert "window.open(" in response.text
    assert "URLSearchParams" in response.text
    assert "Bookmarklet" in response.text  # heading visible


def test_bookmarklet_uses_public_base_url_when_set(
    http_client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from vocab_api.config import settings

    monkeypatch.setattr(settings, "public_base_url", "https://vocab.example.com")
    response = http_client.get("/bookmarklet", headers=_alice())
    assert response.status_code == 200
    assert "https://vocab.example.com/?" in response.text
