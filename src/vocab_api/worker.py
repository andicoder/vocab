import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .anki_writer import AnkiBackend
from .audio import AudioRequest, AudioStorage, TtsClient, synthesize_with_cache
from .cloze import mask_word_in_sentence
from .gemini import GeminiClient, TranslationRequest, join_collocations, translate_with_cache
from .models import Entry, User
from .operations import ApprovalDeps, write_entry_to_anki

log = logging.getLogger(__name__)

# Cap the worker's exponential backoff at 5 minutes. After ~7 consecutive
# failures with base=5s the unjittered value already exceeds this; further
# failures keep sleeping at the cap (#16).
_BACKOFF_CAP_S = 300.0


def _backoff_seconds(attempt: int, *, base: float, cap: float) -> float:
    """Equal-jitter exponential backoff (AWS-style).

    Returns a random value in [raw/2, raw] where raw = base * 2**(attempt-1)
    capped at `cap`. Equal jitter (rather than full jitter) keeps a useful
    floor so we don't drop back to near-zero sleeps mid-incident, while
    still spreading retries enough to avoid lock-step hammering."""
    raw = min(base * (2 ** (attempt - 1)), cap)
    return random.uniform(raw / 2, raw)


@dataclass(frozen=True, slots=True)
class WorkerDeps:
    """All collaborators an entry needs to flow from `pending` to `synced`.

    Built once at app startup and passed through the worker loop and the
    HTTP request paths so signatures stay short. `cache_session_factory`
    lets translation and audio caches commit independently of the entry
    transaction (see #6)."""

    gemini: GeminiClient
    tts: TtsClient
    storage: AudioStorage
    anki_writer: AnkiBackend
    cache_session_factory: async_sessionmaker[AsyncSession]
    voice: str = "en-US-AriaNeural"


async def process_entry(
    *,
    session: AsyncSession,
    entry: Entry,
    user: User,
    deps: WorkerDeps,
) -> str | None:
    """Translates and (if plausible) writes `entry` to Anki.

    Returns the existing lemma if `entry` was a duplicate of another entry's
    lemma — in that case the entry is deleted from the session and no further
    work happens. Caller may use the return value for user feedback."""
    translation = await translate_with_cache(
        session=session,
        cache_session_factory=deps.cache_session_factory,
        gemini=deps.gemini,
        request=TranslationRequest(word=entry.word, sentence=entry.sentence, lang=entry.lang),
    )

    # Kindle imports often produce two surface forms of one lemma (e.g.
    # "dozens" + "dozen"). Drop the second one before it hits
    # uq_entry_user_lemma_lang_sense at commit time and triggers an infinite
    # worker retry (#10). Since #24 the duplicate guard is per-sense — the
    # same lemma can legitimately land twice as long as the sense_key
    # differs (`train` as verb vs. as noun).
    if await _sense_already_exists(
        session,
        user_id=user.id,
        lemma=translation.lemma,
        lang=entry.lang,
        sense_key=translation.sense_key,
        exclude_id=entry.id,
    ):
        log.info(
            "dropped duplicate-sense entry id=%s user=%s lemma=%s sense=%s",
            entry.id,
            user.id,
            translation.lemma,
            translation.sense_key,
        )
        await session.delete(entry)
        return translation.lemma

    verdict = await deps.gemini.plausibility(
        word=entry.word, sentence=entry.sentence, translation=translation
    )
    audio_url = await synthesize_with_cache(
        session=session,
        cache_session_factory=deps.cache_session_factory,
        tts=deps.tts,
        storage=deps.storage,
        request=AudioRequest(word=translation.lemma, voice=deps.voice, lang=entry.lang),
    )
    await _populate_cloze(entry, gemini=deps.gemini, lemma=translation.lemma)

    entry.lemma = translation.lemma
    entry.translation = translation.translation
    entry.alternatives = translation.alternatives
    entry.ipa = translation.ipa
    entry.sense_key = translation.sense_key
    entry.sense_label = translation.sense_label or None
    entry.collocations = join_collocations(translation.collocations) or None
    entry.audio_url = audio_url

    if verdict == "YES":
        await write_entry_to_anki(
            entry=entry,
            user=user,
            deps=ApprovalDeps(storage=deps.storage, anki_writer=deps.anki_writer, voice=deps.voice),
        )
        log.info(
            "synced entry id=%s user=%s lemma=%s",
            entry.id,
            user.id,
            translation.lemma,
        )
    else:
        entry.status = "needs-review"
        log.info(
            "entry needs review id=%s user=%s lemma=%s verdict=%s",
            entry.id,
            user.id,
            translation.lemma,
            verdict,
        )
    return None


async def _populate_cloze(entry: Entry, *, gemini: GeminiClient, lemma: str) -> None:
    """Fill `entry.cloze_sentence` (and `entry.sentence` if it was empty).

    Prefers the deterministic path: mask the user-submitted surface form
    (entry.word) inside the user-submitted source sentence. Falls back to
    a Gemini-invented example only when there is no source sentence at all
    or when the surface form does not appear in the source sentence
    (a rare edge case worth a warning)."""
    if entry.sentence:
        masked = mask_word_in_sentence(word=entry.word, sentence=entry.sentence)
        if masked is not None:
            entry.cloze_sentence = masked
            return
        log.warning(
            "cloze regex miss id=%s word=%r — falling back to invented example",
            entry.id,
            entry.word,
        )

    invented = await gemini.invent_example(lemma=lemma)
    entry.sentence = invented.sentence
    entry.cloze_sentence = invented.cloze_sentence


async def _sense_already_exists(  # noqa: PLR0913 — six narrow scalars are clearer than a one-call-site dataclass
    session: AsyncSession,
    *,
    user_id: int,
    lemma: str,
    lang: str,
    sense_key: str,
    exclude_id: int,
) -> bool:
    stmt = (
        select(Entry.id)
        .where(
            Entry.user_id == user_id,
            Entry.lemma == lemma,
            Entry.lang == lang,
            Entry.sense_key == sense_key,
            Entry.id != exclude_id,
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _claim_one_pending(session: AsyncSession) -> Entry | None:
    # `with_for_update(skip_locked=True)` lets multiple worker replicas race
    # for entries without blocking each other; whoever claims the row first
    # processes it.
    stmt = (
        select(Entry)
        .where(Entry.status == "pending")
        .order_by(Entry.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


@asynccontextmanager
async def run_worker(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    deps: WorkerDeps,
    poll_interval_s: float = 5.0,
    throttle_s: float = 1.0,
) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(
        _worker_loop(
            session_factory=session_factory,
            deps=deps,
            poll_interval_s=poll_interval_s,
            throttle_s=throttle_s,
        ),
        name="vocab-worker",
    )
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _worker_loop(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    deps: WorkerDeps,
    poll_interval_s: float,
    throttle_s: float,
) -> None:
    consecutive_failures = 0
    while True:
        try:
            processed = await _process_one(session_factory=session_factory, deps=deps)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker iteration failed")
            consecutive_failures += 1
            backoff = _backoff_seconds(
                consecutive_failures, base=poll_interval_s, cap=_BACKOFF_CAP_S
            )
            log.warning(
                "worker backoff %.1fs after %s consecutive errors",
                backoff,
                consecutive_failures,
            )
            await asyncio.sleep(backoff)
            continue

        consecutive_failures = 0
        # Throttle when work was found to spread load; back off harder when
        # the queue is empty to avoid hammering Postgres for no reason.
        await asyncio.sleep(throttle_s if processed else poll_interval_s)


async def _process_one(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    deps: WorkerDeps,
) -> bool:
    async with session_factory() as session, session.begin():
        entry = await _claim_one_pending(session)
        if entry is None:
            return False
        user = await session.get(User, entry.user_id)
        assert user is not None
        await process_entry(session=session, entry=entry, user=user, deps=deps)
    return True
