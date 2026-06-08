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
    "ExtraExamples",
    "ClozeSentence",
    "Translation",
    "Alternatives",
    "IPA",
    "SenseLabel",
    "Collocations",
    "Audio",
    "Source",
    "DateAdded",
    # Idiomatic-alternative block (#60). All five render together inside a
    # single `{{#AltLemma}}…{{/AltLemma}}` conditional so cards without an
    # alternative look exactly like before.
    "AltLemma",
    "AltReason",
    "AltTranslation",
    "AltIPA",
    "AltExamples",
    "AltAudio",
]

CardDirection = Literal["de_en", "en_de", "both"]

# Template names match how Anki users see them in the notetype manager —
# direct, not generic "Card 1" / "Card 2" which makes mixed-direction
# collections unreadable.
_NAME_DE_EN = "DE→EN"
_NAME_EN_DE = "EN→DE"

# German display names per source language. New entries land under
# `{LANGUAGE}::{DIRECTION}` so reviews can be paced per (lang, direction).
# Unknown codes fall back to the uppercase ISO tag (e.g. `tlh → TLH::…`)
# rather than crashing the writer.
_LANG_DISPLAY_NAMES_DE = {
    "en": "Englisch",
    "es": "Spanisch",
    "nl": "Niederländisch",
    "fr": "Französisch",
    "it": "Italienisch",
    "pt": "Portugiesisch",
    "hr": "Kroatisch",
}


def deck_name_for(*, lang: str, template_name: str) -> str:
    parent = _LANG_DISPLAY_NAMES_DE.get(lang, lang.upper())
    return f"{parent}::{template_name}"


# Semantic HTML so Anki's default CSS already produces a readable
# hierarchy (no shipped CSS in the notetype; users can layer their own
# styles through the notetype's Styling tab and `_refresh_template`
# never touches `model['css']`). No section labels — visual cues
# (quotes, italics, small caps, `<hr>`) carry the structure (#46).
_FRONT_DE_EN = (
    "<p>{{ClozeSentence}}</p>"
    "<p><small>"
    "({{Translation}}{{#SenseLabel}}, {{SenseLabel}}{{/SenseLabel}})"
    "</small></p>"
)
_ALT_BLOCK = (
    "{{#AltLemma}}"
    "<hr>"
    "<p><small><em>{{AltReason}} — more common: <strong>{{AltLemma}}</strong>"
    " {{#AltIPA}}<small>{{AltIPA}}</small>{{/AltIPA}}"
    " — {{AltTranslation}}</em></small></p>"
    "{{#AltAudio}}<p>{{AltAudio}}</p>{{/AltAudio}}"
    "{{#AltExamples}}<p><em>{{AltExamples}}</em></p>{{/AltExamples}}"
    "{{/AltLemma}}"
)
_BACK_DE_EN = (
    "{{FrontSide}}<hr>"
    "<h2>{{Lemma}}</h2>"
    "{{#IPA}}<small>{{IPA}}</small>{{/IPA}}"
    "{{#Audio}}<p>{{Audio}}</p>{{/Audio}}"
    "<p><strong>{{Translation}}</strong>"
    "{{#SenseLabel}} <em>({{SenseLabel}})</em>{{/SenseLabel}}"
    "</p>"
    "{{#Alternatives}}<p><small><em>{{Alternatives}}</em></small></p>{{/Alternatives}}"
    '<p><em>„{{Sentence}}"</em></p>'
    "{{#ExtraExamples}}<p>{{ExtraExamples}}</p>{{/ExtraExamples}}"
    "{{#Collocations}}<p><small>{{Collocations}}</small></p>{{/Collocations}}"
    + _ALT_BLOCK
    + "{{#Source}}<p><small>{{Source}}</small></p>{{/Source}}"
)

# Recognition direction: lemma + IPA on the front, translation + supporting
# context on the back. The headword is the dictionary form, not the user's
# inflected input — the surface form still appears in the example sentence.
# Audio on the back so listening always follows the recall attempt.
_FRONT_EN_DE = "<h2>{{Lemma}}</h2>{{#IPA}}<small>{{IPA}}</small>{{/IPA}}"
_BACK_EN_DE = (
    "{{FrontSide}}<hr>"
    "<p><strong>{{Translation}}</strong>"
    "{{#SenseLabel}} <em>({{SenseLabel}})</em>{{/SenseLabel}}"
    "</p>"
    "{{Audio}}"
    "{{#Alternatives}}<p><small><em>{{Alternatives}}</em></small></p>{{/Alternatives}}"
    '<p><em>„{{Sentence}}"</em></p>'
    "{{#ExtraExamples}}<p>{{ExtraExamples}}</p>{{/ExtraExamples}}"
    "{{#Collocations}}<p><small>{{Collocations}}</small></p>{{/Collocations}}" + _ALT_BLOCK
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
    collocations: str
    extra_examples: str
    audio_data: bytes | None
    audio_filename: str | None
    source: str | None
    # Idiomatic-alternative payload (#60). Empty strings (and None for the
    # audio bytes/filename) mean "no alternative" — the conditional template
    # block hides itself in that case.
    alt_lemma: str = ""
    alt_reason: str = ""
    alt_translation: str = ""
    alt_ipa: str = ""
    alt_examples: str = ""
    alt_audio_data: bytes | None = None
    alt_audio_filename: str | None = None


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
        lang: str = "en",
    ) -> int: ...

    async def update_card(self, *, username: str, card_id: int, cloze_sentence: str) -> None: ...


def write_media_file(media_dir: Path, content: VocabCardContent) -> None:
    if content.audio_data is not None and content.audio_filename:
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / content.audio_filename).write_bytes(content.audio_data)
    if content.alt_audio_data is not None and content.alt_audio_filename:
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / content.alt_audio_filename).write_bytes(content.alt_audio_data)


def add_vocab_note(
    col: Collection,
    content: VocabCardContent,
    *,
    direction: CardDirection = "de_en",
    lang: str = "en",
) -> int:
    """Adds a Vocab note to `col` and returns the resulting card id.

    Caller is responsible for the Collection lifecycle (open/close) and
    for placing audio media bytes in the collection's media directory."""
    model = _ensure_notetype(col, direction=direction, lang=lang)
    # Each template owns its `did`; the note's own deck_id only matters as
    # a fallback for templates without one. Use the first template's deck
    # so the note has a sensible "home" in Anki's browse view.
    home_deck_id = model["tmpls"][0]["did"]

    note = col.new_note(model)
    note["Word"] = content.word
    note["Lemma"] = content.lemma
    note["Sentence"] = content.sentence or ""
    note["ClozeSentence"] = content.cloze_sentence
    note["Translation"] = content.translation
    note["Alternatives"] = content.alternatives
    note["IPA"] = content.ipa
    note["SenseLabel"] = content.sense_label
    note["Collocations"] = content.collocations
    note["ExtraExamples"] = content.extra_examples
    note["Audio"] = f"[sound:{content.audio_filename}]" if content.audio_filename else ""
    note["Source"] = content.source or ""
    note["DateAdded"] = datetime.now(UTC).date().isoformat()
    note["AltLemma"] = content.alt_lemma
    note["AltReason"] = content.alt_reason
    note["AltTranslation"] = content.alt_translation
    note["AltIPA"] = content.alt_ipa
    note["AltExamples"] = content.alt_examples
    note["AltAudio"] = f"[sound:{content.alt_audio_filename}]" if content.alt_audio_filename else ""

    col.add_note(note, deck_id=home_deck_id)
    card_ids = col.find_cards(f"nid:{note.id}")
    return int(card_ids[0])


def update_vocab_note(col: Collection, card_id: int, cloze_sentence: str) -> None:
    """Update the ClozeSentence field of an existing Anki note.

    Only touches the notes table — card scheduling (due, interval, ease) is
    stored separately in the cards table and is not affected (#82)."""
    from anki.cards import CardId  # noqa: PLC0415 — avoid circular at module level
    from anki.notes import NoteId  # noqa: PLC0415 — avoid circular at module level

    card = col.get_card(CardId(card_id))
    note = col.get_note(NoteId(card.nid))
    note["ClozeSentence"] = cloze_sentence
    col.update_note(note)


class AnkiWriter:
    """File-based backend: writes notes straight into `<root>/<user>/collection.anki2`.

    Used in tests/dev. In production it conflicts with anki-sync-server
    holding the same file open — switch to AnkiSyncWriter there."""

    def __init__(self, *, root: Path) -> None:
        self._root = root

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
        lang: str = "en",
    ) -> int:
        # The anki package opens a SQLite Collection synchronously through a
        # Rust backend; running it on the event loop would block other I/O.
        return await asyncio.to_thread(self._write_sync, username, content, direction, lang)

    async def update_card(self, *, username: str, card_id: int, cloze_sentence: str) -> None:
        await asyncio.to_thread(self._update_sync, username, card_id, cloze_sentence)

    def _update_sync(self, username: str, card_id: int, cloze_sentence: str) -> None:
        col_path = self.collection_path(username)
        col = Collection(str(col_path))
        try:
            update_vocab_note(col, card_id, cloze_sentence)
        finally:
            col.close()

    def _write_sync(
        self, username: str, content: VocabCardContent, direction: CardDirection, lang: str
    ) -> int:
        col_path = self.collection_path(username)
        col_path.parent.mkdir(parents=True, exist_ok=True)
        write_media_file(self.media_dir(username), content)

        col = Collection(str(col_path))
        try:
            return add_vocab_note(col, content, direction=direction, lang=lang)
        finally:
            col.close()


def _ensure_notetype(
    col: Collection, *, direction: CardDirection = "de_en", lang: str = "en"
) -> dict[str, Any]:
    existing = col.models.by_name(VOCAB_NOTETYPE)
    if existing is None:
        return _create_notetype(col, direction=direction, lang=lang)
    _migrate_notetype(col, existing, direction=direction, lang=lang)
    return existing


def _create_notetype(col: Collection, *, direction: CardDirection, lang: str) -> dict[str, Any]:
    model = col.models.new(VOCAB_NOTETYPE)
    for field_name in VOCAB_FIELDS:
        col.models.add_field(model, col.models.new_field(field_name))

    for name, qfmt, afmt in _wanted_templates(direction):
        template = col.models.new_template(name)
        template["qfmt"] = qfmt
        template["afmt"] = afmt
        template["did"] = col.decks.id(deck_name_for(lang=lang, template_name=name))
        col.models.add_template(model, template)
    col.models.add(model)
    return model


def _migrate_notetype(
    col: Collection, model: dict[str, Any], *, direction: CardDirection, lang: str
) -> None:
    """Bring an existing 'Vocab' notetype up to the current field+template spec.

    Idempotent — called on every write. Templates the user no longer wants
    are KEPT (removing them would delete every card generated from that
    template in their collection); direction can only be expanded safely."""
    changed = _add_missing_fields(col, model)
    changed = _rename_legacy_templates(model) or changed
    changed = _sync_templates(col, model, direction=direction, lang=lang) or changed
    if changed:
        col.models.update_dict(model)


def _add_missing_fields(col: Collection, model: dict[str, Any]) -> bool:
    existing = {f["name"] for f in model["flds"]}
    added = False
    for field_name in VOCAB_FIELDS:
        if field_name not in existing:
            col.models.add_field(model, col.models.new_field(field_name))
            added = True
    return added


def _rename_legacy_templates(model: dict[str, Any]) -> bool:
    # Pre-#25 collections name the production template "Card 1". Renaming is
    # safe (cards stay attached) and gives a consistent look across freshly
    # created and migrated notetypes.
    renamed = False
    for template in model["tmpls"]:
        if template["name"] == "Card 1":
            template["name"] = _NAME_DE_EN
            renamed = True
    return renamed


def _sync_templates(
    col: Collection, model: dict[str, Any], *, direction: CardDirection, lang: str
) -> bool:
    existing_names = {t["name"] for t in model["tmpls"]}
    changed = False
    for name, qfmt, afmt in _wanted_templates(direction):
        # `col.decks.id()` creates the deck if it doesn't exist and only
        # returns None on hard failure; pin the type for downstream helpers.
        deck_id = col.decks.id(deck_name_for(lang=lang, template_name=name))
        assert deck_id is not None
        if name in existing_names:
            changed = _refresh_template(model, name, qfmt, afmt, deck_id) or changed
        else:
            _add_template(col, model, name, qfmt, afmt, deck_id)
            changed = True
    return changed


def _refresh_template(  # noqa: PLR0913 — five small scalars are clearer here than a tuple
    model: dict[str, Any], name: str, qfmt: str, afmt: str, deck_id: int
) -> bool:
    for tmpl in model["tmpls"]:
        if tmpl["name"] != name:
            continue
        changed = False
        if tmpl["qfmt"] != qfmt or tmpl["afmt"] != afmt:
            tmpl["qfmt"] = qfmt
            tmpl["afmt"] = afmt
            changed = True
        # Always re-pin `did` so a language switch on the user's account
        # routes future cards into the new subdeck. Existing cards keep their
        # deck — Anki sets the card's deck at creation time.
        if tmpl.get("did") != deck_id:
            tmpl["did"] = deck_id
            changed = True
        return changed
    return False


def _add_template(  # noqa: PLR0913 — five small scalars are clearer here than a tuple
    col: Collection, model: dict[str, Any], name: str, qfmt: str, afmt: str, deck_id: int
) -> None:
    tmpl = col.models.new_template(name)
    tmpl["qfmt"] = qfmt
    tmpl["afmt"] = afmt
    tmpl["did"] = deck_id
    col.models.add_template(model, tmpl)
