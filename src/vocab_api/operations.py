from datetime import UTC, datetime

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .anki_writer import AnkiWriter
from .audio import AudioStorage, audio_key
from .models import Entry, User


class ApprovePayload(BaseModel):
    lemma: str | None = None
    translation: str | None = None
    alternatives: str | None = None
    ipa: str | None = None


async def load_owned_entry(session: AsyncSession, entry_id: int, user: User) -> Entry:
    entry = await session.get(Entry, entry_id)
    if entry is None or entry.user_id != user.id:
        raise HTTPException(status_code=404, detail="entry not found")
    return entry


async def apply_approve(
    *,
    entry: Entry,
    payload: ApprovePayload,
    user: User,
    storage: AudioStorage,
    anki_writer: AnkiWriter,
    voice: str,
) -> None:
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
        audio_filename = audio_key(entry.lemma, voice, entry.lang)
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


def apply_reject(entry: Entry) -> None:
    entry.status = "rejected"
