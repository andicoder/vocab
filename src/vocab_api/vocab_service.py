import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .gemini import GeminiClient, TranslationResult
from .models import Entry, User
from .operations import ApprovalDeps, ApprovePayload, apply_approve, apply_reject, load_owned_entry
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
