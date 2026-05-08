import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .anki_writer import AnkiWriter
from .audio import AudioRequest, AudioStorage, TtsClient, synthesize_with_cache
from .gemini import GeminiClient, TranslationRequest, translate_with_cache
from .models import Entry, User
from .operations import ApprovalDeps, write_entry_to_anki

log = logging.getLogger(__name__)


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
    anki_writer: AnkiWriter
    cache_session_factory: async_sessionmaker[AsyncSession]
    voice: str = "en-US-AriaNeural"


async def process_entry(
    *,
    session: AsyncSession,
    entry: Entry,
    user: User,
    deps: WorkerDeps,
) -> None:
    translation = await translate_with_cache(
        session=session,
        cache_session_factory=deps.cache_session_factory,
        gemini=deps.gemini,
        request=TranslationRequest(word=entry.word, sentence=entry.sentence, lang=entry.lang),
    )
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

    entry.lemma = translation.lemma
    entry.translation = translation.translation
    entry.alternatives = translation.alternatives
    entry.ipa = translation.ipa
    entry.audio_url = audio_url

    if verdict == "YES":
        await write_entry_to_anki(
            entry=entry,
            user=user,
            deps=ApprovalDeps(storage=deps.storage, anki_writer=deps.anki_writer, voice=deps.voice),
        )
    else:
        entry.status = "needs-review"


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
    while True:
        try:
            processed = await _process_one(session_factory=session_factory, deps=deps)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker iteration failed")
            await asyncio.sleep(poll_interval_s)
            continue

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
