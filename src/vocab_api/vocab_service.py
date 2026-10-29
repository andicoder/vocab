from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .anki_writer import AnkiBackend
from .exercise import exercise_pool
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
from .worker import WorkerDeps, process_entry_or_defer


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

    duplicate_of = await process_entry_or_defer(
        session=session, entry=entry, user=user, deps=deps, timeout=timeout
    )

    await session.commit()
    # Duplicate sense: process_entry deleted the new row, so adding is
    # idempotent — hand the caller the entry that already exists.
    if duplicate_of is not None:
        await session.refresh(duplicate_of)
        return duplicate_of
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
    """Advance cloze_index for every synced entry with something to rotate.

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
        # An entry rotates when either pool has something to cycle: more
        # than one gap sentence, or more than one exercise type.
        if len(cloze_pool(entry)) <= 1 and len(exercise_pool(entry)) <= 1:
            continue
        await rotate_cloze(entry=entry, anki_writer=anki_writer, user=user)
        await session.commit()
        rotated += 1

    return rotated
