import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .anki_writer import AnkiBackend
from .gemini import GeminiClient, TranslationResult
from .models import Entry, User
from .operations import (
    ApprovalDeps,
    ApprovePayload,
    apply_approve,
    apply_reject,
    cloze_pool,
    load_owned_entry,
    rotate_cloze,
)
from .schemas import EntryCreate
from .worker import WorkerDeps, process_entry


async def add_entry(
    *,
    session: AsyncSession,
    user: User,
    payload: EntryCreate,
    deps: WorkerDeps,
    timeout: float = 10.0,
) -> Entry:
    entry = Entry(
        user_id=user.id,
        word=payload.word,
        sentence=payload.sentence,
        source=payload.source,
        lang=payload.lang,
    )
    session.add(entry)
    await session.flush()

    try:
        async with asyncio.timeout(timeout):
            await process_entry(session=session, entry=entry, user=user, deps=deps)
    except (TimeoutError, httpx.HTTPError):
        pass

    await session.commit()
    await session.refresh(entry)
    return entry


async def list_entries(
    *,
    session: AsyncSession,
    user: User,
    status_filter: str | None = None,
    limit: int = 100,
) -> list[Entry]:
    stmt = select(Entry).where(Entry.user_id == user.id).order_by(Entry.created_at.desc())
    if status_filter:
        stmt = stmt.where(Entry.status == status_filter)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def approve_entry(
    *,
    session: AsyncSession,
    entry_id: int,
    user: User,
    payload: ApprovePayload,
    deps: ApprovalDeps,
) -> Entry:
    entry = await load_owned_entry(session, entry_id, user)
    await apply_approve(session=session, entry=entry, payload=payload, user=user, deps=deps)
    await session.commit()
    await session.refresh(entry)
    return entry


async def reject_entry(
    *,
    session: AsyncSession,
    entry_id: int,
    user: User,
) -> Entry:
    entry = await load_owned_entry(session, entry_id, user)
    apply_reject(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def translate_word(
    *,
    gemini: GeminiClient,
    word: str,
    sentence: str | None = None,
) -> TranslationResult:
    return await gemini.translate(word=word, sentence=sentence)


async def rotate_cloze_sentences(
    *,
    session: AsyncSession,
    user: User,
    anki_writer: AnkiBackend,
) -> int:
    """Advance cloze_index for every synced entry whose pool has >1 sentence.

    Returns the number of entries rotated."""
    stmt = (
        select(Entry)
        .where(Entry.user_id == user.id, Entry.status == "synced", Entry.anki_card_id.is_not(None))
        .order_by(Entry.id)
    )
    result = await session.execute(stmt)
    entries = list(result.scalars().all())

    rotated = 0
    for entry in entries:
        if len(cloze_pool(entry)) <= 1:
            continue
        await rotate_cloze(entry=entry, anki_writer=anki_writer, user=user)
        await session.commit()
        rotated += 1

    return rotated
