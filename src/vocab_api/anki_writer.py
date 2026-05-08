import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from anki.collection import Collection

VOCAB_NOTETYPE = "Vocab"
VOCAB_FIELDS = [
    "Word",
    "Lemma",
    "Sentence",
    "Translation",
    "Alternatives",
    "IPA",
    "Audio",
    "Source",
    "DateAdded",
]


@dataclass(frozen=True, slots=True)
class VocabCardContent:
    """All per-card fields that go into the Anki note."""

    word: str
    lemma: str
    sentence: str | None
    translation: str
    alternatives: str
    ipa: str
    audio_data: bytes | None
    audio_filename: str | None
    source: str | None


class AnkiBackend(Protocol):
    """Anything that can persist a vocab card for a given user.

    Two implementations exist: `AnkiWriter` opens the user's
    `collection.anki2` directly on disk (simple, but conflicts with a
    long-running anki-sync-server holding the same file — see #5).
    `AnkiSyncWriter` instead opens a private shadow collection and pushes
    via the Anki sync HTTP protocol."""

    async def write_card(self, *, username: str, content: VocabCardContent) -> int: ...


def write_media_file(media_dir: Path, content: VocabCardContent) -> None:
    if content.audio_data is not None and content.audio_filename:
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / content.audio_filename).write_bytes(content.audio_data)


def add_vocab_note(col: Collection, deck_name: str, content: VocabCardContent) -> int:
    """Adds a Vocab note to `col` and returns the resulting card id.

    Caller is responsible for the Collection lifecycle (open/close) and
    for placing audio media bytes in the collection's media directory."""
    model = _ensure_notetype(col)
    deck_id = col.decks.id(deck_name)
    assert deck_id is not None

    note = col.new_note(model)
    note["Word"] = content.word
    note["Lemma"] = content.lemma
    note["Sentence"] = content.sentence or ""
    note["Translation"] = content.translation
    note["Alternatives"] = content.alternatives
    note["IPA"] = content.ipa
    note["Audio"] = f"[sound:{content.audio_filename}]" if content.audio_filename else ""
    note["Source"] = content.source or ""
    note["DateAdded"] = datetime.now(UTC).date().isoformat()

    col.add_note(note, deck_id=deck_id)
    card_ids = col.find_cards(f"nid:{note.id}")
    return int(card_ids[0])


class AnkiWriter:
    """File-based backend: writes notes straight into `<root>/<user>/collection.anki2`.

    Used in tests/dev. In production it conflicts with anki-sync-server
    holding the same file open — switch to AnkiSyncWriter there."""

    def __init__(self, *, root: Path, deck_name: str = "Default") -> None:
        self._root = root
        self._deck_name = deck_name

    def collection_path(self, username: str) -> Path:
        return self._root / username / "collection.anki2"

    def media_dir(self, username: str) -> Path:
        return self._root / username / "collection.media"

    async def write_card(self, *, username: str, content: VocabCardContent) -> int:
        # The anki package opens a SQLite Collection synchronously through a
        # Rust backend; running it on the event loop would block other I/O.
        return await asyncio.to_thread(self._write_sync, username, content)

    def _write_sync(self, username: str, content: VocabCardContent) -> int:
        col_path = self.collection_path(username)
        col_path.parent.mkdir(parents=True, exist_ok=True)
        write_media_file(self.media_dir(username), content)

        col = Collection(str(col_path))
        try:
            return add_vocab_note(col, self._deck_name, content)
        finally:
            col.close()


def _ensure_notetype(col: Collection) -> dict:  # type: ignore[type-arg]
    existing = col.models.by_name(VOCAB_NOTETYPE)
    if existing is not None:
        return existing

    model = col.models.new(VOCAB_NOTETYPE)
    for field_name in VOCAB_FIELDS:
        col.models.add_field(model, col.models.new_field(field_name))

    template = col.models.new_template("Card 1")
    template["qfmt"] = (
        '<div class="word">{{Word}}</div><div class="sentence">{{Sentence}}</div>{{Audio}}'
    )
    template["afmt"] = (
        "{{FrontSide}}<hr>"
        '<div class="translation">{{Translation}}</div>'
        '<div class="alternatives">{{Alternatives}}</div>'
        '<div class="ipa">{{IPA}}</div>'
        '<div class="source">{{Source}}</div>'
    )
    col.models.add_template(model, template)
    col.models.add(model)
    return model
