import asyncio
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..anki_writer import AnkiBackend
from ..audio import AudioStorage, TtsClient
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..deps import (
    get_anki_writer,
    get_gemini,
    get_session_factory,
    get_storage,
    get_tts,
)
from ..gemini import GeminiClient
from ..models import Entry, User
from ..operations import (
    ApprovalDeps,
    ApprovePayload,
    apply_approve,
    apply_reject,
    load_owned_entry,
)
from ..schemas import EntryCreate, EntryRead
from ..worker import WorkerDeps, process_entry

router = APIRouter(prefix="/vocab", tags=["vocab"])


@router.post("", response_model=EntryRead, status_code=status.HTTP_201_CREATED)
async def create_entry(
    payload: EntryCreate,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
    tts: Annotated[TtsClient, Depends(get_tts)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
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

    if settings.gemini_api_key:
        deps = WorkerDeps(
            gemini=gemini,
            tts=tts,
            storage=storage,
            anki_writer=anki_writer,
            cache_session_factory=session_factory,
            voice=settings.audio_voice,
        )
        try:
            async with asyncio.timeout(settings.gemini_timeout_s):
                await process_entry(session=session, entry=entry, user=user, deps=deps)
        except (TimeoutError, httpx.HTTPError):
            pass

    await session.commit()
    await session.refresh(entry)
    return entry


@router.get("", response_model=list[EntryRead])
async def list_entries(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[Entry]:
    stmt = select(Entry).where(Entry.user_id == user.id).order_by(Entry.created_at.desc())
    if status_filter:
        stmt = stmt.where(Entry.status == status_filter)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


@router.post("/{entry_id}/approve", response_model=EntryRead)
async def approve_entry(
    entry_id: int,
    payload: ApprovePayload,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
) -> Entry:
    entry = await load_owned_entry(session, entry_id, user)
    await apply_approve(
        entry=entry,
        payload=payload,
        user=user,
        deps=ApprovalDeps(
            storage=storage,
            anki_writer=anki_writer,
            gemini=gemini,
            voice=settings.audio_voice,
        ),
    )
    await session.commit()
    await session.refresh(entry)
    return entry


@router.post("/{entry_id}/reject", response_model=EntryRead)
async def reject_entry(
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Entry:
    entry = await load_owned_entry(session, entry_id, user)
    apply_reject(entry)
    await session.commit()
    await session.refresh(entry)
    return entry
