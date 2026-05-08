from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..audio import (
    AudioStorage,
    LocalDirAudioStorage,
    TtsClient,
    audio_key,
    synthesize_with_cache,
)
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..deps import get_session_factory, get_storage, get_tts
from ..models import User

router = APIRouter(tags=["audio"])


@router.get("/audio/{word}.mp3")
async def live_audio(
    word: str,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    tts: Annotated[TtsClient, Depends(get_tts)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> Response:
    voice = settings.audio_voice
    lang = "en"
    await synthesize_with_cache(
        session=session,
        cache_session_factory=session_factory,
        tts=tts,
        storage=storage,
        word=word,
        voice=voice,
        lang=lang,
    )
    await session.commit()

    key = audio_key(word, voice, lang)
    if isinstance(storage, LocalDirAudioStorage):
        return FileResponse(storage.root / key, media_type="audio/mpeg")
    data = await storage.fetch(key)
    return Response(content=data, media_type="audio/mpeg")
