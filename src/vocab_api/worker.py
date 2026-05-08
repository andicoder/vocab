import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .anki_writer import AnkiWriter
from .audio import AudioStorage, TtsClient, synthesize_with_cache
from .gemini import GeminiClient, translate_with_cache
from .models import Entry, User
from .operations import write_entry_to_anki

log = logging.getLogger(__name__)


async def process_entry(
    *,
    session: AsyncSession,
    entry: Entry,
    user: User,
    gemini: GeminiClient,
    tts: TtsClient,
    storage: AudioStorage,
    anki_writer: AnkiWriter,
    voice: str = "en-US-AriaNeural",
) -> None:
    translation = await translate_with_cache(
        session=session,
        gemini=gemini,
        word=entry.word,
        sentence=entry.sentence,
        lang=entry.lang,
    )
    verdict = await gemini.plausibility(
        word=entry.word, sentence=entry.sentence, translation=translation
    )
    audio_url = await synthesize_with_cache(
        session=session,
        tts=tts,
        storage=storage,
        word=translation.lemma,
        voice=voice,
        lang=entry.lang,
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
            storage=storage,
            anki_writer=anki_writer,
            voice=voice,
        )
    else:
        entry.status = "needs-review"


async def _claim_one_pending(session: AsyncSession) -> Entry | None:
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
    gemini: GeminiClient,
    tts: TtsClient,
    storage: AudioStorage,
    anki_writer: AnkiWriter,
    voice: str = "en-US-AriaNeural",
    poll_interval_s: float = 5.0,
    throttle_s: float = 1.0,
) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(
        _worker_loop(
            session_factory=session_factory,
            gemini=gemini,
            tts=tts,
            storage=storage,
            anki_writer=anki_writer,
            voice=voice,
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
    gemini: GeminiClient,
    tts: TtsClient,
    storage: AudioStorage,
    anki_writer: AnkiWriter,
    voice: str,
    poll_interval_s: float,
    throttle_s: float,
) -> None:
    while True:
        try:
            processed = await _process_one(
                session_factory=session_factory,
                gemini=gemini,
                tts=tts,
                storage=storage,
                anki_writer=anki_writer,
                voice=voice,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker iteration failed")
            await asyncio.sleep(poll_interval_s)
            continue

        await asyncio.sleep(throttle_s if processed else poll_interval_s)


async def _process_one(
    *,
    session_factory: async_sessionmaker[AsyncSession],
    gemini: GeminiClient,
    tts: TtsClient,
    storage: AudioStorage,
    anki_writer: AnkiWriter,
    voice: str,
) -> bool:
    async with session_factory() as session, session.begin():
        entry = await _claim_one_pending(session)
        if entry is None:
            return False
        user = await session.get(User, entry.user_id)
        assert user is not None
        await process_entry(
            session=session,
            entry=entry,
            user=user,
            gemini=gemini,
            tts=tts,
            storage=storage,
            anki_writer=anki_writer,
            voice=voice,
        )
    return True
