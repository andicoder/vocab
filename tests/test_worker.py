import json

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from vocab_api.gemini import GeminiClient
from vocab_api.models import Entry, User
from vocab_api.worker import process_entry


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

    def public_url(self, key: str) -> str:
        return f"https://cdn.example.com/{key}"


async def _make_pending_entry(session: AsyncSession, **overrides) -> Entry:
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
    return entry


_TRANSLATE_PAYLOAD = {
    "lemma": "expedition",
    "translation": "die Expedition",
    "alternatives": "die Reise",
    "ipa": "/ˌɛkspɪˈdɪʃən/",
}


async def test_process_entry_yes_auto_approves(db_session: AsyncSession):
    entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session, entry=entry, gemini=gemini, tts=tts, storage=storage
        )
    finally:
        await http.aclose()

    assert entry.status == "auto-approved"
    assert entry.lemma == "expedition"
    assert entry.translation == "die Expedition"
    assert entry.alternatives == "die Reise"
    assert entry.ipa == "/ˌɛkspɪˈdɪʃən/"
    assert entry.audio_url is not None
    assert entry.audio_url.startswith("https://cdn.example.com/")
    assert tts.calls == [("expedition", "en-US-AriaNeural")]


async def test_process_entry_unclear_needs_review(db_session: AsyncSession):
    entry = await _make_pending_entry(db_session, word="bank")
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
            session=db_session, entry=entry, gemini=gemini, tts=tts, storage=storage
        )
    finally:
        await http.aclose()

    assert entry.status == "needs-review"
    assert entry.translation == "die Bank"
    assert entry.audio_url is not None


async def test_process_entry_no_needs_review(db_session: AsyncSession):
    entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "NO"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session, entry=entry, gemini=gemini, tts=tts, storage=storage
        )
    finally:
        await http.aclose()

    assert entry.status == "needs-review"


async def test_process_entry_uses_lemma_for_audio(db_session: AsyncSession):
    """Audio is keyed by lemma so 'expeditions' and 'expedition' share one MP3."""
    entry = await _make_pending_entry(db_session, word="expeditions")
    gemini, http = _gemini_client(_make_handler(_TRANSLATE_PAYLOAD, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session, entry=entry, gemini=gemini, tts=tts, storage=storage
        )
    finally:
        await http.aclose()

    assert tts.calls == [("expedition", "en-US-AriaNeural")]


async def test_process_entry_propagates_translation_error(db_session: AsyncSession):
    entry = await _make_pending_entry(db_session)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    gemini, http = _gemini_client(handler)
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        with pytest.raises(httpx.HTTPStatusError):
            await process_entry(
                session=db_session, entry=entry, gemini=gemini, tts=tts, storage=storage
            )
    finally:
        await http.aclose()

    assert entry.status == "pending"
    assert entry.translation is None
