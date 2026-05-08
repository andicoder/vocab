import asyncio
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..anki_writer import AnkiWriter
from ..audio import AudioStorage, TtsClient, audio_key
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..deps import get_anki_writer, get_gemini, get_storage, get_tts
from ..gemini import GeminiClient
from ..models import Entry, User
from ..schemas import EntryCreate, EntryRead
from ..worker import process_entry

router = APIRouter(prefix="/vocab", tags=["vocab"])


class ApprovePayload(BaseModel):
    lemma: str | None = None
    translation: str | None = None
    alternatives: str | None = None
    ipa: str | None = None


@router.post("", response_model=EntryRead, status_code=status.HTTP_201_CREATED)
async def create_entry(
    payload: EntryCreate,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
    tts: Annotated[TtsClient, Depends(get_tts)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
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
        try:
            async with asyncio.timeout(settings.gemini_timeout_s):
                await process_entry(
                    session=session,
                    entry=entry,
                    gemini=gemini,
                    tts=tts,
                    storage=storage,
                    voice=settings.audio_voice,
                )
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


async def _load_owned_entry(session: AsyncSession, entry_id: int, user: User) -> Entry:
    entry = await session.get(Entry, entry_id)
    if entry is None or entry.user_id != user.id:
        raise HTTPException(status_code=404, detail="entry not found")
    return entry


@router.post("/{entry_id}/approve", response_model=EntryRead)
async def approve_entry(
    entry_id: int,
    payload: ApprovePayload,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiWriter, Depends(get_anki_writer)],
) -> Entry:
    entry = await _load_owned_entry(session, entry_id, user)

    if payload.lemma is not None:
        entry.lemma = payload.lemma
    if payload.translation is not None:
        entry.translation = payload.translation
    if payload.alternatives is not None:
        entry.alternatives = payload.alternatives
    if payload.ipa is not None:
        entry.ipa = payload.ipa

    if not entry.lemma or not entry.translation:
        raise HTTPException(status_code=400, detail="entry not yet translated")

    audio_data: bytes | None = None
    audio_filename: str | None = None
    if entry.audio_url:
        audio_filename = audio_key(entry.lemma, settings.audio_voice, entry.lang)
        audio_data = await storage.fetch(audio_filename)

    card_id = await anki_writer.write_card(
        username=user.username,
        word=entry.word,
        lemma=entry.lemma,
        sentence=entry.sentence,
        translation=entry.translation,
        alternatives=entry.alternatives or "",
        ipa=entry.ipa or "",
        audio_data=audio_data,
        audio_filename=audio_filename,
        source=entry.source,
    )

    now = datetime.now(UTC)
    entry.anki_card_id = card_id
    entry.status = "synced"
    entry.approved_at = now
    entry.synced_at = now

    await session.commit()
    await session.refresh(entry)
    return entry


@router.post("/{entry_id}/reject", response_model=EntryRead)
async def reject_entry(
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Entry:
    entry = await _load_owned_entry(session, entry_id, user)
    entry.status = "rejected"
    await session.commit()
    await session.refresh(entry)
    return entry
