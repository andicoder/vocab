import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from anki.collection import Collection

VOCAB_NOTETYPE = "Vocab"
VOCAB_FIELDS = [
    "Word",
    "Lemma",
    "Sentence",
    "ClozeSentence",
    "Translation",
    "Alternatives",
    "IPA",
    "SenseLabel",
    "Audio",
    "Source",
    "DateAdded",
]

CardDirection = Literal["de_en", "en_de", "both"]

# Template names match how Anki users see them in the notetype manager —
# direct, not generic "Card 1" / "Card 2" which makes mixed-direction
# collections unreadable.
_NAME_DE_EN = "DE→EN"
_NAME_EN_DE = "EN→DE"

# Anki's conditional field rendering `{{#SenseLabel}}…{{/SenseLabel}}` shows
# the inner block only when SenseLabel is non-empty — so monosemous words
# render as "(die Bank)" rather than "(die Bank, )".
_FRONT_DE_EN = (
    '<div class="cloze-sentence">{{ClozeSentence}}</div>'
    '<div class="hint">({{Translation}}{{#SenseLabel}}, {{SenseLabel}}{{/SenseLabel}})</div>'
)
_BACK_DE_EN = (
    "{{FrontSide}}<hr>"
    '<div class="word">{{Word}}</div>'
    "{{Audio}}"
    '<div class="sentence">{{Sentence}}</div>'
    '<div class="alternatives">{{Alternatives}}</div>'
    '<div class="ipa">{{IPA}}</div>'
    '<div class="source">{{Source}}</div>'
)

# Recognition direction: bare word on the front, German translation +
# everything else on the back. Audio on the back too, so listening
# always follows the recall attempt regardless of direction.
_FRONT_EN_DE = '<div class="word">{{Word}}</div>'
_BACK_EN_DE = (
    "{{FrontSide}}<hr>"
    '<div class="translation">{{Translation}}{{#SenseLabel}}, {{SenseLabel}}{{/SenseLabel}}</div>'
    "{{Audio}}"
    '<div class="sentence">{{Sentence}}</div>'
    '<div class="alternatives">{{Alternatives}}</div>'
    '<div class="ipa">{{IPA}}</div>'
)


def _wanted_templates(direction: CardDirection) -> list[tuple[str, str, str]]:
    """Return (name, qfmt, afmt) tuples in render order for a given direction."""
    de_en = (_NAME_DE_EN, _FRONT_DE_EN, _BACK_DE_EN)
    en_de = (_NAME_EN_DE, _FRONT_EN_DE, _BACK_EN_DE)
    if direction == "de_en":
        return [de_en]
    if direction == "en_de":
        return [en_de]
    return [de_en, en_de]


@dataclass(frozen=True, slots=True)
class VocabCardContent:
    """All per-card fields that go into the Anki note."""

    word: str
    lemma: str
    sentence: str | None
    cloze_sentence: str
    translation: str
    alternatives: str
    ipa: str
    sense_label: str
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

    async def write_card(
        self,
        *,
        username: str,
        content: VocabCardContent,
        direction: CardDirection = "de_en",
    ) -> int: ...


def write_media_file(media_dir: Path, content: VocabCardContent) -> None:
    if content.audio_data is not None and content.audio_filename:
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / content.audio_filename).write_bytes(content.audio_data)


def add_vocab_note(
    col: Collection,
    deck_name: str,
    content: VocabCardContent,
    *,
    direction: CardDirection = "de_en",
) -> int:
    """Adds a Vocab note to `col` and returns the resulting card id.

    Caller is responsible for the Collection lifecycle (open/close) and
    for placing audio media bytes in the collection's media directory."""
    model = _ensure_notetype(col, direction=direction)
    deck_id = col.decks.id(deck_name)
    assert deck_id is not None

    note = col.new_note(model)
    note["Word"] = content.word
    note["Lemma"] = content.lemma
    note["Sentence"] = content.sentence or ""
    note["ClozeSentence"] = content.cloze_sentence
    note["Translation"] = content.translation
    note["Alternatives"] = content.alternatives
    note["IPA"] = content.ipa
    note["SenseLabel"] = content.sense_label
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

    async def write_card(
        self,
        *,
        username: str,
        content: VocabCardContent,
        direction: CardDirection = "de_en",
    ) -> int:
        # The anki package opens a SQLite Collection synchronously through a
        # Rust backend; running it on the event loop would block other I/O.
        return await asyncio.to_thread(self._write_sync, username, content, direction)

    def _write_sync(
        self, username: str, content: VocabCardContent, direction: CardDirection
    ) -> int:
        col_path = self.collection_path(username)
        col_path.parent.mkdir(parents=True, exist_ok=True)
        write_media_file(self.media_dir(username), content)

        col = Collection(str(col_path))
        try:
            return add_vocab_note(col, self._deck_name, content, direction=direction)
        finally:
            col.close()


def _ensure_notetype(
    col: Collection, *, direction: CardDirection = "de_en"
) -> dict[str, Any]:
    existing = col.models.by_name(VOCAB_NOTETYPE)
    if existing is None:
        return _create_notetype(col, direction=direction)
    _migrate_notetype(col, existing, direction=direction)
    return existing


def _create_notetype(col: Collection, *, direction: CardDirection) -> dict[str, Any]:
    model = col.models.new(VOCAB_NOTETYPE)
    for field_name in VOCAB_FIELDS:
        col.models.add_field(model, col.models.new_field(field_name))

    for name, qfmt, afmt in _wanted_templates(direction):
        template = col.models.new_template(name)
        template["qfmt"] = qfmt
        template["afmt"] = afmt
        col.models.add_template(model, template)
    col.models.add(model)
    return model


def _migrate_notetype(
    col: Collection, model: dict[str, Any], *, direction: CardDirection
) -> None:
    """Bring an existing 'Vocab' notetype up to the current field+template spec.

    Called on every write, so it must be idempotent and cheap when there's
    nothing to do. Missing fields get appended (Anki forbids removing fields
    that already hold data, and we don't need to). Templates that the user
    wants but don't exist yet get added; templates the user no longer wants
    are KEPT — removing them would delete every card generated from that
    template in the user's collection. Direction can only be expanded
    safely; shrinking requires a manual edit in Anki's notetype manager."""
    existing_field_names = {f["name"] for f in model["flds"]}
    added_field = False
    for field_name in VOCAB_FIELDS:
        if field_name not in existing_field_names:
            col.models.add_field(model, col.models.new_field(field_name))
            added_field = True

    # Pre-#25 collections name the production template "Card 1". Renaming is
    # safe (cards stay attached) and gives a consistent look across freshly
    # created and migrated notetypes.
    legacy_renamed = False
    for template in model["tmpls"]:
        if template["name"] == "Card 1":
            template["name"] = _NAME_DE_EN
            legacy_renamed = True

    wanted = _wanted_templates(direction)
    existing_template_names = {t["name"] for t in model["tmpls"]}
    template_added = False
    template_refreshed = False

    for name, qfmt, afmt in wanted:
        if name in existing_template_names:
            for tmpl in model["tmpls"]:
                if tmpl["name"] == name and (tmpl["qfmt"] != qfmt or tmpl["afmt"] != afmt):
                    tmpl["qfmt"] = qfmt
                    tmpl["afmt"] = afmt
                    template_refreshed = True
        else:
            tmpl = col.models.new_template(name)
            tmpl["qfmt"] = qfmt
            tmpl["afmt"] = afmt
            col.models.add_template(model, tmpl)
            template_added = True

    if added_field or legacy_renamed or template_added or template_refreshed:
        col.models.update_dict(model)
