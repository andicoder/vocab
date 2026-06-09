from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .anki_writer import AnkiBackend, CardDirection, VocabCardContent
from .audio import AudioRequest, AudioStorage, TtsClient, audio_key, synthesize_with_cache
from .cloze import mask_word_in_sentence, populate_cloze
from .gemini import (
    GeminiClient,
    TranslationRequest,
    TranslationResult,
    join_collocations,
    join_extra_examples,
    translate_with_cache,
)
from .kindle import parse_kindle_vocab
from .models import Entry, User


class ApprovePayload(BaseModel):
    lemma: str | None = None
    translation: str | None = None
    alternatives: str | None = None
    ipa: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovalDeps:
    """Collaborators needed to finalize an entry into an Anki card.

    `gemini` + `cache_session_factory` only fire inside `apply_approve` to
    backfill worker-only fields (cloze_sentence per #57; extra_examples,
    collocations, sense_label per #58; alt_* per #60) on legacy entries
    that landed in `needs-review` before the worker populated them. Fresh
    entries already have those fields set.

    `tts` is needed alongside `storage` so that pre-#60 entries which now
    backfill an `alt_lemma` can also synthesize the alternative's audio at
    approve time — `synthesize_with_cache` short-circuits on a cache hit,
    so the cost is bounded."""

    storage: AudioStorage
    anki_writer: AnkiBackend
    gemini: GeminiClient
    tts: TtsClient
    cache_session_factory: async_sessionmaker[AsyncSession]
    voice: str


def cloze_pool(entry: Entry) -> list[str]:
    """Candidate cloze sentences: cloze_sentence plus masked extra examples.

    Extra examples are stored as full sentences; each one is masked with the
    lemma before it enters the pool. If the lemma isn't found, the original
    surface form is tried next. Sentences where neither form is found are
    silently dropped — a smaller pool is better than exposing the answer on
    the card front."""
    pool = [entry.cloze_sentence or ""]
    if entry.extra_examples and entry.lemma:
        for s in entry.extra_examples.split("<br>"):
            if not s:
                continue
            masked = mask_word_in_sentence(word=entry.lemma, sentence=s)
            if masked is None and entry.word != entry.lemma:
                masked = mask_word_in_sentence(word=entry.word, sentence=s)
            if masked is not None:
                pool.append(masked)
    return pool


def active_cloze_sentence(entry: Entry) -> str:
    """The sentence currently active based on cloze_index (wraps around the pool)."""
    pool = cloze_pool(entry)
    return pool[entry.cloze_index % len(pool)]


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

    alt_audio_data: bytes | None = None
    alt_audio_filename: str | None = None
    if entry.alt_audio_url and entry.alt_lemma:
        alt_audio_filename = audio_key(
            AudioRequest(word=entry.alt_lemma, voice=deps.voice, lang=entry.lang)
        )
        alt_audio_data = await deps.storage.fetch(alt_audio_filename)

    content = VocabCardContent(
        word=entry.word,
        lemma=entry.lemma,
        sentence=entry.sentence,
        cloze_sentence=active_cloze_sentence(entry),
        translation=entry.translation,
        alternatives=entry.alternatives or "",
        ipa=entry.ipa or "",
        sense_label=entry.sense_label or "",
        collocations=entry.collocations or "",
        extra_examples=entry.extra_examples or "",
        audio_data=audio_data,
        audio_filename=audio_filename,
        source=entry.source,
        alt_lemma=entry.alt_lemma or "",
        alt_reason=entry.alt_reason or "",
        alt_translation=entry.alt_translation or "",
        alt_ipa=entry.alt_ipa or "",
        alt_examples=entry.alt_examples or "",
        alt_audio_data=alt_audio_data,
        alt_audio_filename=alt_audio_filename,
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
    session: AsyncSession,
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

    if not entry.cloze_sentence:
        await populate_cloze(entry, gemini=deps.gemini, lemma=entry.lemma)

    await _backfill_worker_fields(session=session, entry=entry, deps=deps)

    await write_entry_to_anki(entry=entry, user=user, deps=deps)


async def _backfill_worker_fields(
    *, session: AsyncSession, entry: Entry, deps: ApprovalDeps
) -> None:
    """Re-run translation for legacy entries missing worker-only fields.

    Pre-#23 needs-review rows never went through `worker.process_entry`,
    and pre-#60 rows never had the idiomatic-alternative scoring run on
    them. The triggering signal is `alt_priority`: post-#60 worker writes
    always set it (to ``"preferred"``/``"minor"``/``"none"``), so a NULL
    means the row is legacy and needs backfilling for every worker-only
    field at once (#58, #60).

    The cache hit path is cheap for any word the worker has already seen —
    a miss falls through to one Gemini call. Fresh entries arrive here
    with `alt_priority` set and short-circuit before the lookup."""
    if entry.alt_priority is not None:
        return
    tr = await translate_with_cache(
        session=session,
        cache_session_factory=deps.cache_session_factory,
        gemini=deps.gemini,
        request=TranslationRequest(word=entry.word, sentence=entry.sentence, lang=entry.lang),
    )
    _backfill_translation_fields(entry, tr)
    if entry.alt_lemma and not entry.alt_audio_url:
        entry.alt_audio_url = await synthesize_with_cache(
            session=session,
            cache_session_factory=deps.cache_session_factory,
            tts=deps.tts,
            storage=deps.storage,
            request=AudioRequest(word=entry.alt_lemma, voice=deps.voice, lang=entry.lang),
        )


def _backfill_translation_fields(entry: Entry, tr: TranslationResult) -> None:
    """Copy missing scalar/list fields from a fresh translation onto `entry`.

    Pure column copy — no I/O. Pulled out so `_backfill_worker_fields` can
    stay below ruff's C901 complexity budget; the audio synthesis (and the
    early-return guard) live in the caller."""
    if not entry.extra_examples:
        entry.extra_examples = join_extra_examples(tr.extra_examples) or None
    if not entry.collocations:
        entry.collocations = join_collocations(tr.collocations) or None
    if not entry.sense_label:
        entry.sense_label = tr.sense_label or None
    if not entry.alt_lemma:
        entry.alt_lemma = tr.alt_lemma or None
    if not entry.alt_reason:
        entry.alt_reason = tr.alt_reason or None
    if not entry.alt_translation:
        entry.alt_translation = tr.alt_translation or None
    if not entry.alt_ipa:
        entry.alt_ipa = tr.alt_ipa or None
    if not entry.alt_examples:
        entry.alt_examples = join_extra_examples(tr.alt_examples) or None
    entry.alt_priority = tr.alt_priority or None


async def rotate_cloze(*, entry: Entry, anki_writer: AnkiBackend, user: User) -> None:
    """Advance cloze_index by one and push the new sentence to Anki."""
    assert entry.anki_card_id is not None
    entry.cloze_index += 1
    await anki_writer.update_card(
        username=user.username,
        card_id=entry.anki_card_id,
        cloze_sentence=active_cloze_sentence(entry),
    )


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
