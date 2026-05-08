import json

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vocab_api.db import SessionLocal
from vocab_api.gemini import GeminiClient, TranslationResult, translate_with_cache
from vocab_api.models import TranslationCache


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def _payload(translation: str = "die Expedition") -> str:
    return json.dumps(
        {
            "lemma": "expedition",
            "translation": translation,
            "alternatives": "",
            "ipa": "/ˌɛkspɪˈdɪʃən/",
        }
    )


def _client(handler) -> tuple[GeminiClient, httpx.AsyncClient]:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return (
        GeminiClient(
            http=http,
            api_key="k",
            model="gemini-2.5-flash-lite",
            base_url="https://example.com/v1beta",
        ),
        http,
    )


async def test_cache_miss_calls_gemini_and_stores(db_session: AsyncSession):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_gemini_response(_payload()))

    gemini, http = _client(handler)
    try:
        result = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="expedition",
            sentence="A grand expedition north.",
        )
    finally:
        await http.aclose()

    assert calls == 1
    assert result == TranslationResult(
        lemma="expedition",
        translation="die Expedition",
        alternatives="",
        ipa="/ˌɛkspɪˈdɪʃən/",
    )

    rows = (await db_session.execute(select(TranslationCache))).scalars().all()
    assert len(rows) == 1
    assert rows[0].word == "expedition"
    assert rows[0].translation == "die Expedition"
    assert rows[0].sentence_hash is not None


async def test_cache_hit_skips_gemini(db_session: AsyncSession):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_gemini_response(_payload()))

    gemini, http = _client(handler)
    try:
        await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="expedition",
            sentence="A grand expedition north.",
        )
        result = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="expedition",
            sentence="A grand expedition north.",
        )
    finally:
        await http.aclose()

    assert calls == 1
    assert result.translation == "die Expedition"


async def test_cache_distinguishes_by_sentence(db_session: AsyncSession):
    payloads = iter([_payload("die Bank"), _payload("das Ufer")])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(next(payloads)))

    gemini, http = _client(handler)
    try:
        first = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="bank",
            sentence="I deposited money at the bank.",
        )
        second = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="bank",
            sentence="We sat on the river bank.",
        )
    finally:
        await http.aclose()

    assert first.translation == "die Bank"
    assert second.translation == "das Ufer"

    rows = (await db_session.execute(select(TranslationCache))).scalars().all()
    assert len(rows) == 2


async def test_cache_with_no_sentence_uses_null_hash(db_session: AsyncSession):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_gemini_response(_payload()))

    gemini, http = _client(handler)
    try:
        await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="expedition",
            sentence=None,
        )
        await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            word="expedition",
            sentence=None,
        )
    finally:
        await http.aclose()

    assert calls == 1
    rows = (await db_session.execute(select(TranslationCache))).scalars().all()
    assert len(rows) == 1
    assert rows[0].sentence_hash is None
