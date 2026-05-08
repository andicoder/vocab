import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vocab_api.anki_writer import AnkiWriter
from vocab_api.audio import AudioRequest, audio_key
from vocab_api.db import SessionLocal
from vocab_api.gemini import GeminiClient
from vocab_api.models import AudioCache, Entry, TranslationCache, User
from vocab_api.worker import WorkerDeps, process_entry


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def _make_handler(translate_payload: dict, verdict: str):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        is_translate = (
            body.get("generationConfig", {}).get("responseMimeType") == "application/json"
        )
        text = json.dumps(translate_payload) if is_translate else verdict
        return httpx.Response(200, json=_gemini_response(text))

    return handler


def _make_translate_only_handler(translate_payload: dict):
    # Asserts plausibility is never called — used to prove the duplicate path
    # short-circuits before the (uncached, expensive) plausibility request.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        is_translate = (
            body.get("generationConfig", {}).get("responseMimeType") == "application/json"
        )
        if not is_translate:
            raise AssertionError("plausibility must not run for duplicates")
        return httpx.Response(200, json=_gemini_response(json.dumps(translate_payload)))

    return handler


def _gemini_client(handler) -> tuple[GeminiClient, httpx.AsyncClient]:
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
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, *, text: str, voice: str) -> bytes:
        self.calls.append((text, voice))
        return b"mp3:" + text.encode()


class _FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    async def fetch(self, key: str) -> bytes:
        return self.objects[key]

    def public_url(self, key: str) -> str:
        return f"https://cdn.example.com/{key}"


async def _make_pending_entry(session: AsyncSession, **overrides) -> tuple[User, Entry]:
    user = User(username="alice")
    session.add(user)
    await session.flush()
    entry = Entry(
        user_id=user.id,
        word=overrides.get("word", "expedition"),
        sentence=overrides.get("sentence", "A grand expedition north."),
        source=overrides.get("source", "test"),
        lang=overrides.get("lang", "en"),
    )
    session.add(entry)
    await session.flush()
    return user, entry


_TRANSLATE_PAYLOAD = {
    "lemma": "expedition",
    "translation": "die Expedition",
    "alternatives": "die Reise",
    "ipa": "/ˌɛkspɪˈdɪʃən/",
}


def _deps(
    *,
    tmp_path: Path,
    gemini: GeminiClient,
    tts: _FakeTts,
    storage: _FakeStorage,
    anki_writer: AnkiWriter | None = None,
) -> WorkerDeps:
    return WorkerDeps(
        gemini=gemini,
        tts=tts,
        storage=storage,
        anki_writer=anki_writer or AnkiWriter(root=tmp_path),
        cache_session_factory=SessionLocal,
    )


async def test_process_entry_yes_writes_anki_and_marks_synced(
    db_session: AsyncSession, tmp_path: Path
):
    user, entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=entry,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert entry.status == "synced"
    assert entry.anki_card_id is not None
    assert entry.approved_at is not None
    assert entry.synced_at is not None
    assert entry.lemma == "expedition"
    assert entry.translation == "die Expedition"
    assert entry.alternatives == "die Reise"
    assert entry.ipa == "/ˌɛkspɪˈdɪʃən/"
    assert entry.audio_url is not None
    assert entry.audio_url.startswith("https://cdn.example.com/")
    assert tts.calls == [("expedition", "en-US-AriaNeural")]
    assert (tmp_path / "alice" / "collection.anki2").exists()
    expected_audio = audio_key(AudioRequest(word="expedition"))
    assert (tmp_path / "alice" / "collection.media" / expected_audio).exists()


async def test_process_entry_unclear_needs_review(db_session: AsyncSession, tmp_path: Path):
    user, entry = await _make_pending_entry(db_session, word="bank")
    gemini, http = _gemini_client(
        _make_handler(
            {
                "lemma": "bank",
                "translation": "die Bank",
                "alternatives": "",
                "ipa": "/bæŋk/",
            },
            "UNCLEAR",
        )
    )
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=entry,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert entry.status == "needs-review"
    assert entry.translation == "die Bank"
    assert entry.audio_url is not None
    assert entry.anki_card_id is None
    assert not (tmp_path / "alice" / "collection.anki2").exists()


async def test_process_entry_no_needs_review(db_session: AsyncSession, tmp_path: Path):
    user, entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "NO"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=entry,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert entry.status == "needs-review"
    assert entry.anki_card_id is None


async def test_process_entry_uses_lemma_for_audio(db_session: AsyncSession, tmp_path: Path):
    # Audio is keyed by lemma so 'expeditions' and 'expedition' share one MP3.
    user, entry = await _make_pending_entry(db_session, word="expeditions")
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=entry,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert tts.calls == [("expedition", "en-US-AriaNeural")]


async def test_process_entry_propagates_translation_error(db_session: AsyncSession, tmp_path: Path):
    user, entry = await _make_pending_entry(db_session)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    gemini, http = _gemini_client(handler)
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        with pytest.raises(httpx.HTTPStatusError):
            await process_entry(
                session=db_session,
                entry=entry,
                user=user,
                deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
            )
    finally:
        await http.aclose()

    assert entry.status == "pending"
    assert entry.translation is None


class _RaisingAnkiWriter:
    def __init__(self) -> None:
        self.calls = 0

    async def write_card(self, **kwargs: object) -> int:
        self.calls += 1
        raise RuntimeError("Anki already open, or media currently syncing")


async def test_process_entry_deletes_entry_when_user_has_existing_lemma(
    db_session: AsyncSession, tmp_path: Path
):
    # Regression for #10: Kindle imports often produce two surface forms for one
    # lemma (e.g. "dozens" and "dozen"). The second one must be dropped instead
    # of hitting uq_entry_user_lemma_lang and leaving the entry pending forever.
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    db_session.add(
        Entry(
            user_id=user.id,
            word="dozen",
            lemma="dozen",
            translation="das Dutzend",
            status="synced",
            lang="en",
        )
    )
    pending = Entry(
        user_id=user.id,
        word="dozens",
        sentence="Dozens of pebbles littered the path.",
        lang="en",
    )
    db_session.add(pending)
    await db_session.flush()
    pending_id = pending.id

    translate_payload = {
        "lemma": "dozen",
        "translation": "das Dutzend",
        "alternatives": "",
        "ipa": "/ˈdʌzən/",
    }
    gemini, http = _gemini_client(_make_translate_only_handler(translate_payload))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        result = await process_entry(
            session=db_session,
            entry=pending,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert result == "dozen"
    await db_session.flush()
    assert await db_session.get(Entry, pending_id) is None
    surviving_count = await db_session.scalar(select(func.count()).select_from(Entry))
    assert surviving_count == 1
    assert tts.calls == []
    assert not (tmp_path / "alice" / "collection.anki2").exists()


async def test_process_entry_allows_same_lemma_for_different_users(
    db_session: AsyncSession, tmp_path: Path
):
    # Duplicate detection is per-user — Alice and Bob can each have "expedition".
    alice = User(username="alice")
    bob = User(username="bob")
    db_session.add_all([alice, bob])
    await db_session.flush()

    db_session.add(
        Entry(
            user_id=alice.id,
            word="expedition",
            lemma="expedition",
            translation="die Expedition",
            status="synced",
            lang="en",
        )
    )
    bob_entry = Entry(
        user_id=bob.id,
        word="expedition",
        sentence="A grand expedition north.",
        lang="en",
    )
    db_session.add(bob_entry)
    await db_session.flush()

    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=bob_entry,
            user=bob,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert bob_entry.status == "synced"
    assert bob_entry.lemma == "expedition"


async def test_caches_survive_anki_write_failure(db_session: AsyncSession, tmp_path: Path):
    user, entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()
    raising_anki = _RaisingAnkiWriter()

    try:
        with pytest.raises(RuntimeError, match="Anki already open"):
            await process_entry(
                session=db_session,
                entry=entry,
                user=user,
                deps=_deps(
                    tmp_path=tmp_path,
                    gemini=gemini,
                    tts=tts,
                    storage=storage,
                    anki_writer=raising_anki,  # type: ignore[arg-type]
                ),
            )
    finally:
        await http.aclose()

    assert raising_anki.calls == 1

    async with SessionLocal() as fresh:
        translation_count = await fresh.scalar(select(func.count()).select_from(TranslationCache))
        audio_count = await fresh.scalar(select(func.count()).select_from(AudioCache))
        assert translation_count == 1
        assert audio_count == 1
