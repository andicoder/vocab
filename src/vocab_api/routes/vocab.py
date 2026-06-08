from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
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
from ..operations import ApprovalDeps, ApprovePayload
from ..schemas import EntryCreate, EntryRead
from ..vocab_service import add_entry, approve_entry, list_entries, reject_entry, rotate_cloze_sentences
from ..worker import WorkerDeps

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
    deps = WorkerDeps(
        gemini=gemini,
        tts=tts,
        storage=storage,
        anki_writer=anki_writer,
        cache_session_factory=session_factory,
        voice=settings.audio_voice,
    )
    return await add_entry(
        session=session,
        user=user,
        payload=payload,
        deps=deps,
        timeout=settings.gemini_timeout_s,
    )


@router.get("", response_model=list[EntryRead])
async def list_entries_route(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[Entry]:
    return await list_entries(session=session, user=user, status_filter=status_filter, limit=limit)


@router.post("/{entry_id}/approve", response_model=EntryRead)
async def approve_entry_route(
    entry_id: int,
    payload: ApprovePayload,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
    tts: Annotated[TtsClient, Depends(get_tts)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> Entry:
    deps = ApprovalDeps(
        storage=storage,
        anki_writer=anki_writer,
        gemini=gemini,
        tts=tts,
        cache_session_factory=session_factory,
        voice=settings.audio_voice,
    )
    return await approve_entry(
        session=session, entry_id=entry_id, user=user, payload=payload, deps=deps
    )


@router.post("/{entry_id}/reject", response_model=EntryRead)
async def reject_entry_route(
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Entry:
    return await reject_entry(session=session, entry_id=entry_id, user=user)


@router.post("/rotate-cloze")
async def rotate_cloze_route(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
) -> dict[str, int]:
    rotated = await rotate_cloze_sentences(session=session, user=user, anki_writer=anki_writer)
    return {"rotated": rotated}
