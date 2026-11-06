import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.llm_fake import make_llm, reply
from tests.test_routes import stub_translation_llm
from vocab_api.anki_writer import AnkiWriter
from vocab_api.audio import LocalDirAudioStorage
from vocab_api.db import engine
from vocab_api.deps import get_anki_writer, get_llm, get_storage, get_tts
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
    from vocab_api import worker
    from vocab_api.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "fake-key")

    async def fake_process_entry(*, session, entry, user, deps):
        from vocab_api.models import Entry

        await session.delete(entry)
        return Entry(user_id=entry.user_id, word="dozen", lemma="dozen", lang="en")

    monkeypatch.setattr(worker, "process_entry", fake_process_entry)

    response = http_client.post(
        "/ui/vocab",
        data={"word": "dozens", "sentence": "Dozens of pebbles."},
        headers=_alice(),
    )
    assert response.status_code == 200
    assert "dozens" in response.text
    assert "dozen" in response.text
    assert "schon" in response.text  # de: "schon als ... in der Sammlung"


def test_htmx_create_entry_keeps_entry_pending_when_tts_fails(
    http_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Same failure as the MCP path: an error outside the timeout/HTTP family
    # (edge-tts 403) must leave the raw entry for the worker, not lose it.
    from vocab_api.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "fake-key")

    class BrokenTts:
        async def synthesize(self, *, text: str, voice: str) -> bytes:
            raise RuntimeError("403, message='Invalid response status'")

    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    app.dependency_overrides[get_llm] = stub_translation_llm
    app.dependency_overrides[get_tts] = lambda: BrokenTts()
    app.dependency_overrides[get_storage] = lambda: storage

    response = http_client.post("/ui/vocab", data={"word": "whistle"}, headers=_alice())
    assert response.status_code == 200
    assert "whistle" in response.text

    async def _status() -> str:
        async with engine.connect() as conn:
            row = await conn.execute(text("SELECT status FROM vocab.entry WHERE word = 'whistle'"))
            return row.scalar_one()

    assert asyncio.run(_status()) == "pending"


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
    app.dependency_overrides[get_llm] = stub_translation_llm

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


def _stub_lookup(**overrides: str) -> None:
    payload = {
        "source_lang": "en",
        "english": "eager",
        "translation": "begierig",
        "alternatives": "",
        "ipa": "/ˈiɡər/",
    }
    payload.update(overrides)

    def handler(request):
        return reply(json.dumps(payload))

    llm, _ = make_llm(handler)
    app.dependency_overrides[get_llm] = lambda: llm


def test_translator_page_requires_auth(http_client: TestClient):
    assert http_client.get("/translate").status_code == 401


def test_translator_page_shows_the_form(http_client: TestClient):
    response = http_client.get("/translate", headers=_alice())

    assert response.status_code == 200
    assert 'hx-post="/ui/translate"' in response.text


def test_lookup_shows_the_translation_and_offers_to_save_the_english_word(
    http_client: TestClient,
):
    _stub_lookup(alternatives="gespannt")

    response = http_client.post(
        "/ui/translate",
        data={"text": "eager", "sentence": "He was eager to go."},
        headers=_alice(),
    )

    assert response.status_code == 200
    assert "begierig" in response.text
    assert "gespannt" in response.text
    assert "/ˈiɡər/" in response.text
    assert 'hx-post="/ui/vocab"' in response.text
    assert 'name="word" value="eager"' in response.text
    assert 'value="He was eager to go."' in response.text


def test_lookup_saves_the_surface_form_so_the_sentence_still_contains_it(
    http_client: TestClient,
):
    # The card's gap comes from masking this word in the sentence; the worker
    # finds the lemma itself.
    _stub_lookup(english="smirk", translation="grinsen")

    response = http_client.post(
        "/ui/translate",
        data={"text": "smirked", "sentence": "He smirked at her."},
        headers=_alice(),
    )

    assert 'name="word" value="smirked"' in response.text
    assert 'value="He smirked at her."' in response.text


def test_german_lookup_saves_the_english_word_without_the_german_sentence(
    http_client: TestClient,
):
    _stub_lookup(source_lang="de", english="butterfly", translation="butterfly")

    response = http_client.post(
        "/ui/translate",
        data={"text": "Schmetterling", "sentence": "Der Schmetterling flog weg."},
        headers=_alice(),
    )

    assert 'name="word" value="butterfly"' in response.text
    assert "Der Schmetterling flog weg." not in response.text


def test_lookup_shows_the_english_word_when_the_model_leaves_the_translation_empty(
    http_client: TestClient,
):
    # Seen in the real run: "Schmetterling" returned only english="butterfly".
    _stub_lookup(source_lang="de", english="butterfly", translation="")

    response = http_client.post("/ui/translate", data={"text": "Schmetterling"}, headers=_alice())

    assert "<h2>butterfly</h2>" in response.text


def test_lookup_does_not_create_an_entry(http_client: TestClient):
    _stub_lookup()

    http_client.post("/ui/translate", data={"text": "unsavedlookup"}, headers=_alice())

    async def _count() -> int:
        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT count(*) FROM vocab.entry WHERE word = 'unsavedlookup'")
            )
            return row.scalar_one()

    assert asyncio.run(_count()) == 0


def test_lookup_of_a_whole_sentence_offers_no_save(http_client: TestClient):
    # No single word to learn, so a card would have nothing to teach.
    _stub_lookup(english="", translation="Er wollte gehen.")

    response = http_client.post(
        "/ui/translate", data={"text": "He wanted to go."}, headers=_alice()
    )

    assert "Er wollte gehen." in response.text
    assert "/ui/vocab" not in response.text


def test_lookup_failure_shows_a_toast(http_client: TestClient):
    def handler(request):
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    llm, _ = make_llm(handler)
    app.dependency_overrides[get_llm] = lambda: llm

    response = http_client.post("/ui/translate", data={"text": "eager"}, headers=_alice())

    assert response.status_code == 200
    assert response.headers["HX-Retarget"] == "#toast"
