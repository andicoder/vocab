import asyncio
import json
import logging
from collections.abc import Iterator
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


_DEFAULT_INVENTED = {
    "sentence": "An invented example sentence.",
    "cloze_sentence": "An invented example ___.",
}


def _classify(body: dict) -> str:
    """Decide which Gemini operation a mock request represents.

    Both translate and invent_example request JSON; only the prompt text
    can distinguish them. Keep the heuristic narrow so prompt edits don't
    silently re-route tests."""
    is_json = body.get("generationConfig", {}).get("responseMimeType") == "application/json"
    if not is_json:
        return "plausibility"
    prompt = body["contents"][0]["parts"][0]["text"]
    return "invent" if "Invent" in prompt else "translate"


def _make_handler(translate_payload: dict, verdict: str, invented: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        kind = _classify(body)
        if kind == "translate":
            text = json.dumps(translate_payload)
        elif kind == "invent":
            text = json.dumps(invented or _DEFAULT_INVENTED)
        else:
            text = verdict
        return httpx.Response(200, json=_gemini_response(text))

    return handler


def _make_translate_only_handler(translate_payload: dict):
    # Asserts plausibility and invent are never called — used to prove the
    # duplicate path short-circuits before any further Gemini request.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        kind = _classify(body)
        if kind != "translate":
            raise AssertionError(f"only translate may run for duplicates; got {kind}")
        return httpx.Response(200, json=_gemini_response(json.dumps(translate_payload)))

    return handler


def _counting_handler(translate_payload: dict, verdict: str, invented: dict | None = None):
    """Same as _make_handler but exposes a per-kind call counter."""
    counts: dict[str, int] = {"translate": 0, "plausibility": 0, "invent": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        kind = _classify(body)
        counts[kind] += 1
        if kind == "translate":
            text = json.dumps(translate_payload)
        elif kind == "invent":
            text = json.dumps(invented or _DEFAULT_INVENTED)
        else:
            text = verdict
        return httpx.Response(200, json=_gemini_response(text))

    return handler, counts


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


async def test_plausibility_compares_lemma_against_translation(
    db_session: AsyncSession, tmp_path: Path
):
    # Regression for #62: the plausibility check used to pass `entry.word`
    # (the inflected encounter, e.g. `pebbles`) against the lemma-form
    # German translation (`der Kiesel`). A grammatically sensitive LLM
    # then saw the number/tense mismatch and returned UNCLEAR/NO, routing
    # the entry to needs-review even though the translation was correct.
    # The fix is to compare lemma-form against lemma-form by passing
    # `translation.lemma` to plausibility.
    user, entry = await _make_pending_entry(
        db_session,
        word="pebbles",
        sentence="Visitors had left pebbles, in keeping with Jewish tradition.",
    )

    plausibility_prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        kind = _classify(body)
        prompt = body["contents"][0]["parts"][0]["text"]
        if kind == "translate":
            text = json.dumps(
                {
                    "lemma": "pebble",
                    "translation": "der Kiesel",
                    "alternatives": "",
                    "ipa": "/ˈpɛbəl/",
                    "collocations": ["throw pebbles"],
                    "extra_examples": ["She skipped pebbles across the lake."],
                    "alt_priority": "none",
                }
            )
        elif kind == "invent":
            text = json.dumps(_DEFAULT_INVENTED)
        else:
            plausibility_prompts.append(prompt)
            text = "YES"
        return httpx.Response(200, json=_gemini_response(text))

    gemini, http = _gemini_client(handler)
    try:
        await process_entry(
            session=db_session,
            entry=entry,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=_FakeTts(), storage=_FakeStorage()),
        )
    finally:
        await http.aclose()

    assert len(plausibility_prompts) == 1, "plausibility should be checked exactly once"
    prompt = plausibility_prompts[0]
    # Anchor with a trailing newline so `pebble` does not match `pebbles`.
    assert "English word: pebble\n" in prompt, (
        f"plausibility must compare lemma-form against translation; got:\n{prompt}"
    )
    assert "English word: pebbles\n" not in prompt, (
        f"plausibility must not see the inflected encounter; got:\n{prompt}"
    )


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


@pytest.fixture
def worker_log_records() -> Iterator[list[logging.LogRecord]]:
    # pytest's caplog does not reliably capture records from the asyncio
    # path our worker runs on; attach a dedicated handler instead.
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("vocab_api.worker")
    handler = _Capture(level=logging.INFO)
    prior_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


async def test_process_entry_logs_synced_outcome(
    db_session: AsyncSession,
    tmp_path: Path,
    worker_log_records: list[logging.LogRecord],
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

    msgs = [r.getMessage() for r in worker_log_records]
    assert any(
        "synced" in m
        and f"id={entry.id}" in m
        and f"user={user.id}" in m
        and "lemma=expedition" in m
        for m in msgs
    ), msgs


async def test_process_entry_logs_duplicate_drop(
    db_session: AsyncSession,
    tmp_path: Path,
    worker_log_records: list[logging.LogRecord],
):
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
        sentence="Dozens of pebbles.",
        lang="en",
    )
    db_session.add(pending)
    await db_session.flush()
    pending_id = pending.id

    payload = {
        "lemma": "dozen",
        "translation": "das Dutzend",
        "alternatives": "",
        "ipa": "/ˈdʌzən/",
    }
    gemini, http = _gemini_client(_make_translate_only_handler(payload))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=pending,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    msgs = [r.getMessage() for r in worker_log_records]
    assert any(
        "duplicate" in m
        and f"id={pending_id}" in m
        and f"user={user.id}" in m
        and "lemma=dozen" in m
        for m in msgs
    ), msgs


async def test_process_entry_logs_needs_review_outcome(
    db_session: AsyncSession,
    tmp_path: Path,
    worker_log_records: list[logging.LogRecord],
):
    user, entry = await _make_pending_entry(db_session, word="bank")
    payload = {
        "lemma": "bank",
        "translation": "die Bank",
        "alternatives": "",
        "ipa": "/bæŋk/",
    }
    gemini, http = _gemini_client(_make_handler(payload, "UNCLEAR"))
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

    msgs = [r.getMessage() for r in worker_log_records]
    assert any(
        "needs review" in m
        and f"id={entry.id}" in m
        and f"user={user.id}" in m
        and "lemma=bank" in m
        and "verdict=UNCLEAR" in m
        for m in msgs
    ), msgs


def test_backoff_seconds_grows_with_attempt():
    from vocab_api.worker import _backoff_seconds

    # Equal-jitter range: [raw/2, raw] with raw = base * 2^(n-1) capped at cap.
    # Sample many times to assert distribution bounds without flakiness.
    samples_1 = [_backoff_seconds(1, base=5.0, cap=300.0) for _ in range(200)]
    samples_3 = [_backoff_seconds(3, base=5.0, cap=300.0) for _ in range(200)]
    samples_5 = [_backoff_seconds(5, base=5.0, cap=300.0) for _ in range(200)]

    assert all(2.5 <= s <= 5.0 for s in samples_1)
    assert all(10.0 <= s <= 20.0 for s in samples_3)  # raw = 5 * 2^2 = 20
    assert all(40.0 <= s <= 80.0 for s in samples_5)  # raw = 5 * 2^4 = 80
    # Sanity: jitter actually spreads — the min and max in 200 samples
    # should not be identical.
    assert min(samples_3) < max(samples_3)


def test_backoff_seconds_capped_for_large_attempt():
    from vocab_api.worker import _backoff_seconds

    samples = [_backoff_seconds(20, base=5.0, cap=300.0) for _ in range(200)]
    assert all(150.0 <= s <= 300.0 for s in samples)
    assert max(samples) <= 300.0


async def test_worker_loop_backs_off_exponentially_on_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    # Drive _worker_loop directly with a stub _process_one that raises three
    # times then succeeds. Capture the asyncio.sleep durations and assert they
    # grow on failures and reset on success.
    from vocab_api import worker as worker_mod

    failures_remaining = [3]

    async def failing_process_one(*, session_factory, deps):
        if failures_remaining[0] > 0:
            failures_remaining[0] -= 1
            raise RuntimeError("simulated transient failure")
        return False  # quiet success: no work

    sleeps: list[float] = []
    cancel_after = [10]  # safety net: stop the loop eventually

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        cancel_after[0] -= 1
        if cancel_after[0] <= 0:
            raise asyncio.CancelledError

    monkeypatch.setattr(worker_mod, "_process_one", failing_process_one)
    monkeypatch.setattr(worker_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await worker_mod._worker_loop(
            session_factory=SessionLocal,
            deps=WorkerDeps(  # type: ignore[arg-type]
                gemini=None,  # unused: failing_process_one ignores deps
                tts=None,
                storage=None,
                anki_writer=None,
                cache_session_factory=SessionLocal,
            ),
            poll_interval_s=5.0,
            throttle_s=1.0,
        )

    failure_sleeps = sleeps[:3]
    # Equal-jitter: attempt n sleeps in [raw/2, raw], raw = 5 * 2^(n-1) capped.
    assert 2.5 <= failure_sleeps[0] <= 5.0
    assert 5.0 <= failure_sleeps[1] <= 10.0
    assert 10.0 <= failure_sleeps[2] <= 20.0
    # The first post-success sleep is the regular empty-queue poll, not
    # backoff — proves the counter resets.
    assert sleeps[3] == 5.0


async def test_cloze_sentence_derived_from_source_sentence_skips_invent_call(
    db_session: AsyncSession, tmp_path: Path
):
    user, entry = await _make_pending_entry(
        db_session, word="expedition", sentence="A grand expedition north."
    )
    handler, counts = _counting_handler(_TRANSLATE_PAYLOAD, "YES")
    gemini, http = _gemini_client(handler)
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

    assert entry.cloze_sentence == "A grand ___ north."
    # Deterministic regex path — Gemini's invent_example must not be called.
    assert counts["invent"] == 0
    # Source sentence stays untouched on the entry.
    assert entry.sentence == "A grand expedition north."


async def test_cloze_sentence_invented_when_entry_has_no_sentence(
    db_session: AsyncSession, tmp_path: Path
):
    user, entry = await _make_pending_entry(db_session, word="expedition", sentence=None)
    invented = {
        "sentence": "We launched an expedition north.",
        "cloze_sentence": "We launched an ___ north.",
    }
    handler, counts = _counting_handler(_TRANSLATE_PAYLOAD, "YES", invented=invented)
    gemini, http = _gemini_client(handler)
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

    assert counts["invent"] == 1
    assert entry.sentence == "We launched an expedition north."
    assert entry.cloze_sentence == "We launched an ___ north."


async def test_cloze_sentence_invented_when_word_not_in_source_sentence(
    db_session: AsyncSession, tmp_path: Path
):
    # Edge case: user submits `word="dozens"` with a sentence that contains
    # only the lemma form `dozen`. The deterministic regex fails, so the
    # worker has to fall back to inventing an example based on the lemma.
    user, entry = await _make_pending_entry(
        db_session, word="dozens", sentence="Only a dozen left."
    )
    invented = {
        "sentence": "Dozens turned up at the door.",
        "cloze_sentence": "___ turned up at the door.",
    }
    payload = {
        "lemma": "dozen",
        "translation": "das Dutzend",
        "alternatives": "",
        "ipa": "/ˈdʌzən/",
    }
    handler, counts = _counting_handler(payload, "YES", invented=invented)
    gemini, http = _gemini_client(handler)
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

    assert counts["invent"] == 1
    assert entry.cloze_sentence == "___ turned up at the door."
    # Source sentence is overwritten because the original did not contain the
    # surface form — keep front and back in sync.
    assert entry.sentence == "Dozens turned up at the door."


async def test_process_entry_persists_joined_extra_examples(
    db_session: AsyncSession, tmp_path: Path
):
    user, entry = await _make_pending_entry(db_session)
    payload = {
        **_TRANSLATE_PAYLOAD,
        "extra_examples": [
            "The Arctic expedition lasted three months.",
            "She joined an expedition to the Amazon.",
        ],
    }
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
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

    assert entry.extra_examples == (
        "The Arctic expedition lasted three months.<br>She joined an expedition to the Amazon."
    )


async def test_process_entry_leaves_extra_examples_null_when_empty(
    db_session: AsyncSession, tmp_path: Path
):
    user, entry = await _make_pending_entry(db_session)
    payload = {**_TRANSLATE_PAYLOAD, "extra_examples": []}
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
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

    assert entry.extra_examples is None


async def test_process_entry_persists_joined_collocations(db_session: AsyncSession, tmp_path: Path):
    user, entry = await _make_pending_entry(db_session)
    payload = {
        **_TRANSLATE_PAYLOAD,
        "collocations": ["go on an expedition", "Arctic expedition", "lead an expedition"],
    }
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
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

    assert entry.collocations == ("go on an expedition · Arctic expedition · lead an expedition")


async def test_process_entry_leaves_collocations_null_when_gemini_returns_empty(
    db_session: AsyncSession, tmp_path: Path
):
    # Function words / adverbs typically have no idiomatic collocations.
    # Storing empty as NULL keeps the column meaningful (NULL = "no
    # collocations for this lemma") rather than rendering an empty block on
    # the card.
    user, entry = await _make_pending_entry(db_session, word="however")
    payload = {
        "lemma": "however",
        "translation": "jedoch",
        "alternatives": "allerdings",
        "ipa": "",
        "collocations": [],
    }
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
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

    assert entry.collocations is None


async def test_polyseme_same_lemma_different_sense_is_kept_as_separate_card(
    db_session: AsyncSession, tmp_path: Path
):
    # User already has "train" as a noun (Eisenbahn). The new entry uses
    # "train" as a verb (sportlich) — different sense_key, so the worker
    # must NOT drop it. Both cards live side-by-side under #24.
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()
    db_session.add(
        Entry(
            user_id=user.id,
            word="train",
            lemma="train",
            translation="der Zug",
            sense_key="noun-railway",
            sense_label="Eisenbahn",
            status="synced",
            lang="en",
        )
    )
    pending = Entry(
        user_id=user.id,
        word="train",
        sentence="I need to train for a marathon.",
        lang="en",
    )
    db_session.add(pending)
    await db_session.flush()

    payload = {
        "lemma": "train",
        "translation": "trainieren",
        "alternatives": "",
        "ipa": "/treɪn/",
        "sense_key": "verb-exercise",
        "sense_label": "sportlich",
    }
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
    tts = _FakeTts()
    storage = _FakeStorage()

    try:
        await process_entry(
            session=db_session,
            entry=pending,
            user=user,
            deps=_deps(tmp_path=tmp_path, gemini=gemini, tts=tts, storage=storage),
        )
    finally:
        await http.aclose()

    assert pending.status == "synced"
    assert pending.sense_key == "verb-exercise"
    assert pending.sense_label == "sportlich"
    surviving = await db_session.scalar(select(func.count()).select_from(Entry))
    assert surviving == 2  # both senses live as separate cards


async def test_polyseme_same_lemma_same_sense_is_dropped(db_session: AsyncSession, tmp_path: Path):
    # Same (lemma, sense_key) as an existing card — a real duplicate. The
    # worker drops the new entry, just as before (#10 still holds for the
    # same-meaning case).
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()
    db_session.add(
        Entry(
            user_id=user.id,
            word="train",
            lemma="train",
            translation="trainieren",
            sense_key="verb-exercise",
            sense_label="sportlich",
            status="synced",
            lang="en",
        )
    )
    pending = Entry(
        user_id=user.id,
        word="train",
        sentence="I need to train for a marathon.",
        lang="en",
    )
    db_session.add(pending)
    await db_session.flush()
    pending_id = pending.id

    payload = {
        "lemma": "train",
        "translation": "trainieren",
        "alternatives": "",
        "ipa": "/treɪn/",
        "sense_key": "verb-exercise",
        "sense_label": "sportlich",
    }
    gemini, http = _gemini_client(_make_translate_only_handler(payload))
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

    assert result == "train"
    await db_session.flush()
    assert await db_session.get(Entry, pending_id) is None
    assert tts.calls == []


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


# ---------------------------------------------------------------------------
# Idiomatic alternative (#60). The LLM emits alt_priority ∈
# {"preferred", "minor", "none"}: "preferred" routes the entry to needs-review
# even when the plausibility verdict is YES; "minor" and "none" follow the
# existing flow. When alt_lemma is non-empty we also synthesize TTS for it so
# the card can offer a listening cue for the better word.
# ---------------------------------------------------------------------------


_ALT_PAYLOAD_PREFERRED = {
    **_TRANSLATE_PAYLOAD,
    "alt_lemma": "exhausted",
    "alt_reason": "dated",
    "alt_translation": "erschöpft",
    "alt_ipa": "/ɪɡˈzɔːstɪd/",
    "alt_examples": [
        "She was exhausted after the hike.",
        "I'm too exhausted to cook tonight.",
    ],
    "alt_priority": "preferred",
}


_ALT_PAYLOAD_MINOR = {
    **_TRANSLATE_PAYLOAD,
    "alt_lemma": "modern",
    "alt_reason": "formal",
    "alt_translation": "modern",
    "alt_ipa": "/ˈmɒdən/",
    "alt_examples": ["A modern approach works better."],
    "alt_priority": "minor",
}


async def test_process_entry_preferred_alt_routes_to_needs_review(
    db_session: AsyncSession, tmp_path: Path
):
    # Even with a YES plausibility verdict, a strong "more common alternative"
    # signal routes the entry to needs-review so I can confirm before the
    # card is written. The alt_* fields are persisted so the review UI can
    # show them.
    user, entry = await _make_pending_entry(db_session, word="weary")
    gemini, http = _gemini_client(_make_handler(_ALT_PAYLOAD_PREFERRED, "YES"))
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
    assert entry.alt_lemma == "exhausted"
    assert entry.alt_reason == "dated"
    assert entry.alt_translation == "erschöpft"
    assert entry.alt_ipa == "/ɪɡˈzɔːstɪd/"
    assert entry.alt_examples == (
        "She was exhausted after the hike.<br>I'm too exhausted to cook tonight."
    )
    assert entry.alt_audio_url is not None
    assert entry.alt_audio_url.startswith("https://cdn.example.com/")
    # Original lemma and its audio also persisted unchanged — the headword
    # on the card stays the encountered word.
    assert entry.lemma == "expedition"
    assert entry.audio_url is not None
    # TTS fires twice: once for the original lemma, once for the alternative.
    assert ("expedition", "en-US-AriaNeural") in tts.calls
    assert ("exhausted", "en-US-AriaNeural") in tts.calls


async def test_process_entry_minor_alt_still_auto_approves(
    db_session: AsyncSession, tmp_path: Path
):
    # A "minor" stylistic alternative is shown on the card but does not stop
    # auto-approval. Card lands in Anki; alt fields are persisted.
    user, entry = await _make_pending_entry(db_session, word="contemporary")
    gemini, http = _gemini_client(_make_handler(_ALT_PAYLOAD_MINOR, "YES"))
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
    assert entry.alt_lemma == "modern"
    assert entry.alt_priority == "minor"
    assert entry.alt_audio_url is not None


async def test_process_entry_no_alt_leaves_alt_fields_null(
    db_session: AsyncSession, tmp_path: Path
):
    # Common case: Gemini reports `alt_priority="none"` (or omits alt fields).
    # The entry's alt_* columns stay NULL so the card template renders
    # unchanged.
    payload = {**_TRANSLATE_PAYLOAD, "alt_priority": "none"}
    user, entry = await _make_pending_entry(db_session)
    gemini, http = _gemini_client(_make_handler(payload, "YES"))
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
    assert entry.alt_lemma is None
    assert entry.alt_reason is None
    assert entry.alt_translation is None
    assert entry.alt_ipa is None
    assert entry.alt_examples is None
    assert entry.alt_audio_url is None
    # Only one TTS call (for the original lemma) — no alt audio synthesized.
    assert tts.calls == [("expedition", "en-US-AriaNeural")]
