from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .anki_writer import AnkiBackend, CardDirection, VocabCardContent
from .audio import AudioRequest, AudioStorage, audio_key
from .kindle import parse_kindle_vocab
from .models import Entry, User


class ApprovePayload(BaseModel):
    lemma: str | None = None
    translation: str | None = None
    alternatives: str | None = None
    ipa: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalDeps:
    """Collaborators needed to finalize an entry into an Anki card."""

    storage: AudioStorage
    anki_writer: AnkiBackend
    voice: str


async def load_owned_entry(session: AsyncSession, entry_id: int, user: User) -> Entry:
    entry = await session.get(Entry, entry_id)
    if entry is None or entry.user_id != user.id:
        raise HTTPException(status_code=404, detail="entry not found")
    return entry


async def write_entry_to_anki(*, entry: Entry, user: User, deps: ApprovalDeps) -> None:
    # Caller (apply_approve / process_entry) is expected to have populated
    # these fields before this function runs.
    assert entry.lemma is not None and entry.translation is not None

    audio_data: bytes | None = None
    audio_filename: str | None = None
    if entry.audio_url:
        audio_filename = audio_key(
            AudioRequest(word=entry.lemma, voice=deps.voice, lang=entry.lang)
        )
        audio_data = await deps.storage.fetch(audio_filename)

    content = VocabCardContent(
        word=entry.word,
        lemma=entry.lemma,
        sentence=entry.sentence,
        cloze_sentence=entry.cloze_sentence or "",
        translation=entry.translation,
        alternatives=entry.alternatives or "",
        ipa=entry.ipa or "",
        sense_label=entry.sense_label or "",
        collocations=entry.collocations or "",
        extra_examples=entry.extra_examples or "",
        audio_data=audio_data,
        audio_filename=audio_filename,
        source=entry.source,
    )
    direction = cast(CardDirection, user.card_direction)
    card_id = await deps.anki_writer.write_card(
        username=user.username, content=content, direction=direction, lang=entry.lang
    )

    now = datetime.now(UTC)
    entry.anki_card_id = card_id
    entry.status = "synced"
    entry.approved_at = now
    entry.synced_at = now


async def apply_approve(
    *,
    entry: Entry,
    payload: ApprovePayload,
    user: User,
    deps: ApprovalDeps,
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

    await write_entry_to_anki(entry=entry, user=user, deps=deps)


def apply_reject(entry: Entry) -> None:
    entry.status = "rejected"


async def import_kindle_entries(
    *,
    session: AsyncSession,
    user: User,
    db_path: Path,
    lang: str = "en",
) -> tuple[int, int]:
    entries = list(parse_kindle_vocab(db_path, lang=lang))

    seen: set[str] = set()
    unique = []
    for entry in entries:
        if entry.word in seen:
            continue
        seen.add(entry.word)
        unique.append(entry)

    if not unique:
        return 0, 0

    words = [e.word for e in unique]
    existing = await session.execute(
        select(Entry.word).where(
            Entry.user_id == user.id, Entry.lang == lang, Entry.word.in_(words)
        )
    )
    existing_words = {row[0] for row in existing.all()}

    added = skipped = 0
    for entry in unique:
        if entry.word in existing_words:
            skipped += 1
            continue
        session.add(
            Entry(
                user_id=user.id,
                word=entry.word,
                sentence=entry.sentence or None,
                source=f"Kindle: {entry.source}" if entry.source else "Kindle",
                lang=lang,
            )
        )
        added += 1

    await session.flush()
    return added, skipped
