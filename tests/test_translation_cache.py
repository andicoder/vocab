import json

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vocab_api.db import SessionLocal
from vocab_api.gemini import (
    GeminiClient,
    TranslationRequest,
    TranslationResult,
    translate_with_cache,
)
from vocab_api.models import TranslationCache


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def _payload(translation: str = "die Expedition") -> str:
    # The cache reader skips rows where both collocations and extra_examples
    # are empty (#43 — stale rows from migration defaults). Tests that rely
    # on cache *hits* therefore have to return at least one of the two from
    # the mock so the row counts as fresh.
    return json.dumps(
        {
            "lemma": "expedition",
            "translation": translation,
            "alternatives": "",
            "ipa": "/ˌɛkspɪˈdɪʃən/",
            "collocations": ["go on an expedition"],
            "extra_examples": ["She joined an expedition to the Amazon."],
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
            request=TranslationRequest(word="expedition", sentence="A grand expedition north."),
        )
    finally:
        await http.aclose()

    assert calls == 1
    assert result == TranslationResult(
        lemma="expedition",
        translation="die Expedition",
        alternatives="",
        ipa="/ˌɛkspɪˈdɪʃən/",
        collocations=["go on an expedition"],
        extra_examples=["She joined an expedition to the Amazon."],
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
            request=TranslationRequest(word="expedition", sentence="A grand expedition north."),
        )
        result = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            request=TranslationRequest(word="expedition", sentence="A grand expedition north."),
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
            request=TranslationRequest(word="bank", sentence="I deposited money at the bank."),
        )
        second = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            request=TranslationRequest(word="bank", sentence="We sat on the river bank."),
        )
    finally:
        await http.aclose()

    assert first.translation == "die Bank"
    assert second.translation == "das Ufer"

    rows = (await db_session.execute(select(TranslationCache))).scalars().all()
    assert len(rows) == 2


async def test_cache_row_with_empty_collocations_and_extras_is_treated_as_miss(
    db_session: AsyncSession,
):
    # Regression for #43: the migrations that added `collocations` and
    # `extra_examples` to translation_cache backfilled existing rows
    # with `''` (NOT NULL DEFAULT ''). Treating such a row as a cache
    # hit would permanently shadow Gemini's new behavior for the
    # affected words. The reader skips these rows so they get
    # overwritten on the next translate.
    #
    # The stale row is committed via a *separate* session: putting it
    # inside the test's outer transaction (db_session) would deadlock
    # the inner `cache_session_factory` connection that
    # translate_with_cache opens to upsert the fresh row — PG
    # unique-constraint waits cross-connection on the uncommitted key.
    async with SessionLocal() as setup, setup.begin():
        setup.add(
            TranslationCache(
                word="expedition",
                sentence_hash=None,
                lang="en",
                lemma="expedition",
                translation="STALE",
                alternatives="",
                ipa="",
                collocations="",
                extra_examples="",
            )
        )

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.dumps(
            {
                "lemma": "expedition",
                "translation": "die Expedition",
                "alternatives": "",
                "ipa": "/ˌɛkspɪˈdɪʃən/",
                "collocations": ["go on an expedition"],
                "extra_examples": ["She joined an expedition to the Amazon."],
            }
        )
        return httpx.Response(200, json=_gemini_response(payload))

    gemini, http = _client(handler)
    try:
        result = await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            request=TranslationRequest(word="expedition", sentence=None),
        )
    finally:
        await http.aclose()

    assert calls == 1, "stale cache row should not have short-circuited Gemini"
    assert result.translation == "die Expedition"
    assert result.collocations == ["go on an expedition"]


async def test_cache_row_with_only_one_field_populated_still_hits(
    db_session: AsyncSession,
):
    # Rows that have at least one of collocations / extra_examples
    # populated are considered fresh — they came from the post-#26/#27
    # worker. Re-querying every time would defeat the cache.
    # Setup uses a separately committed session for the same reason as
    # the test above (cross-connection unique-constraint deadlock).
    async with SessionLocal() as setup, setup.begin():
        setup.add(
            TranslationCache(
                word="however",
                sentence_hash=None,
                lang="en",
                lemma="however",
                translation="jedoch",
                alternatives="allerdings",
                ipa="",
                collocations="",
                extra_examples="However, the plan succeeded.",
            )
        )

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
            request=TranslationRequest(word="however", sentence=None),
        )
    finally:
        await http.aclose()

    assert calls == 0
    assert result.translation == "jedoch"
    assert result.extra_examples == ["However, the plan succeeded."]


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
            request=TranslationRequest(word="expedition", sentence=None),
        )
        await translate_with_cache(
            session=db_session,
            cache_session_factory=SessionLocal,
            gemini=gemini,
            request=TranslationRequest(word="expedition", sentence=None),
        )
    finally:
        await http.aclose()

    assert calls == 1
    rows = (await db_session.execute(select(TranslationCache))).scalars().all()
    assert len(rows) == 1
    assert rows[0].sentence_hash is None
