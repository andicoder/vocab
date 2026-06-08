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
        "alt_lemma": "",
        "alt_reason": "",
        "alt_translation": "",
        "alt_ipa": "",
        "alt_examples": [],
        "alt_priority": "none",
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
                    "  collocations, extra_examples, alt_priority)"
                    " VALUES ('expedition', :sh, 'en', 'expedition',"
                    " 'die Expedition', '', '', 'noun-journey', 'Reise',"
                    " 'go on an expedition · Arctic expedition',"
                    " 'The Arctic expedition lasted three months.', 'none')"
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


def test_approve_backfills_alt_fields_from_translation_cache(
    http_client: TestClient, tmp_path: Path
):
    # Equivalence-class follow-up to #58: pre-#60 needs-review entries lack the
    # alt_* fields. Approving them must backfill from translate_with_cache so
    # the new card already shows the idiomatic-alternative block.
    storage = LocalDirAudioStorage(root=tmp_path / "audio", public_url_base="/audio")
    anki_writer = AnkiWriter(root=tmp_path / "anki")
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_anki_writer] = lambda: anki_writer

    create = http_client.post(
        "/vocab",
        json={"word": "weary", "sentence": "I felt weary after the hike."},
        headers={"X-authentik-username": "alice"},
    )
    assert create.status_code == 201
    entry_id = create.json()["id"]

    from vocab_api.gemini import _sentence_hash  # noqa: PLC0415 — local helper, test-only use

    sh = _sentence_hash("I felt weary after the hike.")
    assert sh is not None

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE vocab.entry SET status='needs-review',"
                    " cloze_sentence='I felt ___ after the hike.',"
                    " extra_examples='legacy', collocations='legacy',"
                    " sense_label='legacy',"
                    " alt_lemma=NULL, alt_reason=NULL, alt_translation=NULL,"
                    " alt_ipa=NULL, alt_examples=NULL, alt_audio_url=NULL"
                    f" WHERE id = {entry_id}"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.translation_cache"
                    " (word, sentence_hash, lang, lemma, translation,"
                    "  alternatives, ipa, sense_key, sense_label,"
                    "  collocations, extra_examples,"
                    "  alt_lemma, alt_reason, alt_translation, alt_ipa,"
                    "  alt_examples, alt_priority)"
                    " VALUES ('weary', :sh, 'en', 'weary', 'müde',"
                    " '', '', 'adj-tired', 'müde',"
                    " 'go weary', 'I grow weary of this.',"
                    " 'exhausted', 'dated', 'erschöpft', '/ɪɡˈzɔːstɪd/',"
                    " 'She was exhausted after the hike.', 'preferred')"
                ),
                {"sh": sh},
            )

    asyncio.run(_seed())

    response = http_client.post(
        f"/vocab/{entry_id}/approve",
        json={"lemma": "weary", "translation": "müde"},
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    card_id = response.json()["anki_card_id"]
    col = Collection(str(tmp_path / "anki" / "alice" / "collection.anki2"))
    try:
        note = col.get_card(card_id).note()
        assert note["AltLemma"] == "exhausted"
        assert note["AltReason"] == "dated"
        assert note["AltTranslation"] == "erschöpft"
        assert note["AltIPA"] == "/ɪɡˈzɔːstɪd/"
        assert note["AltExamples"] == "She was exhausted after the hike."
        # AltAudio: synthesized by approve flow when alt_lemma was backfilled.
        assert note["AltAudio"].startswith("[sound:")
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


class _FakeAnkiWithUpdate:
    """Fake AnkiBackend that records update_card calls."""

    def __init__(self, *, fail_on_card_id: int | None = None) -> None:
        self.fail_on_card_id = fail_on_card_id
        self.updates: list[dict] = []

    async def write_card(self, *, username: str, **kwargs) -> int:
        return 42

    async def update_card(self, *, username: str, card_id: int, cloze_sentence: str) -> None:
        if card_id == self.fail_on_card_id:
            raise RuntimeError("anki update failed")
        self.updates.append(
            {"username": username, "card_id": card_id, "cloze_sentence": cloze_sentence}
        )


def test_rotate_cloze_rotates_synced_entries_and_returns_count(
    http_client: TestClient,
) -> None:
    fake_anki = _FakeAnkiWithUpdate()
    app.dependency_overrides[get_anki_writer] = lambda: fake_anki

    # Seed a synced entry with a multi-sentence pool directly via SQL.
    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO vocab.user (username) VALUES ('alice') ON CONFLICT DO NOTHING")
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, lang)"
                    " SELECT id, 'train', 'train',"
                    "  'The ___ was late.',"
                    "  'A ___ arrived.<br>The ___ left.',"
                    "  'der Zug', 'synced', 42, 'en'"
                    " FROM vocab.user WHERE username = 'alice'"
                )
            )

    asyncio.run(_seed())

    response = http_client.post(
        "/vocab/rotate-cloze",
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"rotated": 1}
    assert len(fake_anki.updates) == 1
    assert fake_anki.updates[0]["card_id"] == 42
    assert fake_anki.updates[0]["cloze_sentence"] == "A ___ arrived."

    async def _cloze_index() -> int:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT cloze_index FROM vocab.entry WHERE anki_card_id = 42")
            )
            return result.scalar_one()

    assert asyncio.run(_cloze_index()) == 1


def test_rotate_cloze_skips_entries_with_single_sentence_pool(
    http_client: TestClient,
) -> None:
    fake_anki = _FakeAnkiWithUpdate()
    app.dependency_overrides[get_anki_writer] = lambda: fake_anki

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO vocab.user (username) VALUES ('alice') ON CONFLICT DO NOTHING")
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, lang)"
                    " SELECT id, 'train', 'train',"
                    "  'The ___ was late.', NULL,"
                    "  'der Zug', 'synced', 42, 'en'"
                    " FROM vocab.user WHERE username = 'alice'"
                )
            )

    asyncio.run(_seed())

    response = http_client.post(
        "/vocab/rotate-cloze",
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"rotated": 0}
    assert fake_anki.updates == []


def test_rotate_cloze_wraps_to_first_sentence(http_client: TestClient) -> None:
    fake_anki = _FakeAnkiWithUpdate()
    app.dependency_overrides[get_anki_writer] = lambda: fake_anki

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO vocab.user (username) VALUES ('alice') ON CONFLICT DO NOTHING")
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, cloze_index, lang)"
                    " SELECT id, 'train', 'train',"
                    "  'The ___ was late.',"
                    "  'A ___ arrived.<br>The ___ left.',"
                    "  'der Zug', 'synced', 42, 2, 'en'"
                    " FROM vocab.user WHERE username = 'alice'"
                )
            )

    asyncio.run(_seed())

    response = http_client.post(
        "/vocab/rotate-cloze",
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"rotated": 1}
    assert fake_anki.updates[0]["cloze_sentence"] == "The ___ was late."


def test_rotate_cloze_does_not_rotate_other_users_entries(http_client: TestClient) -> None:
    fake_anki = _FakeAnkiWithUpdate()
    app.dependency_overrides[get_anki_writer] = lambda: fake_anki

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(text("INSERT INTO vocab.user (username) VALUES ('alice'), ('bob')"))
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, lang)"
                    " SELECT id, username, username,"
                    "  'The ___ was late.',"
                    "  'A ___ arrived.<br>The ___ left.',"
                    "  'der Zug', 'synced',"
                    "  CASE username WHEN 'alice' THEN 42 ELSE 43 END, 'en'"
                    " FROM vocab.user"
                )
            )

    asyncio.run(_seed())

    response = http_client.post(
        "/vocab/rotate-cloze",
        headers={"X-authentik-username": "alice"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"rotated": 1}
    assert [update["card_id"] for update in fake_anki.updates] == [42]


def test_rotate_cloze_commits_successful_entries_before_later_failure(
    http_client: TestClient,
) -> None:
    fake_anki = _FakeAnkiWithUpdate(fail_on_card_id=43)
    app.dependency_overrides[get_anki_writer] = lambda: fake_anki

    async def _seed() -> None:
        async with engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO vocab.user (username) VALUES ('alice') ON CONFLICT DO NOTHING")
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, lang)"
                    " SELECT id, 'train', 'train',"
                    "  'The ___ was late.',"
                    "  'A ___ arrived.<br>The ___ left.',"
                    "  'der Zug', 'synced', 42, 'en'"
                    " FROM vocab.user WHERE username = 'alice'"
                )
            )
            await conn.execute(
                text(
                    "INSERT INTO vocab.entry"
                    " (user_id, word, lemma, cloze_sentence, extra_examples,"
                    "  translation, status, anki_card_id, lang)"
                    " SELECT id, 'bus', 'bus',"
                    "  'The ___ was late.',"
                    "  'A ___ arrived.<br>The ___ left.',"
                    "  'der Bus', 'synced', 43, 'en'"
                    " FROM vocab.user WHERE username = 'alice'"
                )
            )

    async def _cloze_indexes() -> dict[int, int]:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT anki_card_id, cloze_index FROM vocab.entry ORDER BY anki_card_id")
            )
            return dict(result.all())

    asyncio.run(_seed())

    with pytest.raises(RuntimeError, match="anki update failed"):
        http_client.post(
            "/vocab/rotate-cloze",
            headers={"X-authentik-username": "alice"},
        )

    assert asyncio.run(_cloze_indexes()) == {42: 1, 43: 0}


def test_post_me_token_generates_token(http_client: TestClient) -> None:
    response = http_client.post("/me/token", headers={"X-authentik-username": "alice"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "token" in body
    assert len(body["token"]) >= 32


def test_bearer_token_authenticates_user(http_client: TestClient) -> None:
    token_resp = http_client.post("/me/token", headers={"X-authentik-username": "alice"})
    assert token_resp.status_code == 200
    token = token_resp.json()["token"]

    response = http_client.get("/me/settings", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert response.json() == {"card_direction": "de_en"}


def test_bearer_token_invalid_returns_401(http_client: TestClient) -> None:
    response = http_client.get("/me/settings", headers={"Authorization": "Bearer invalid-token"})
    assert response.status_code == 401


def test_cors_preflight_allows_extension_origin(http_client: TestClient) -> None:
    response = http_client.options(
        "/vocab",
        headers={
            "Origin": "moz-extension://abc123-def456",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )
    assert response.status_code == 200, response.text
    assert "access-control-allow-origin" in response.headers
