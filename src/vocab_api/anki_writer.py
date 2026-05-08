import asyncio
from datetime import UTC, datetime
from pathlib import Path

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


class AnkiWriter:
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
        word: str,
        lemma: str,
        sentence: str | None,
        translation: str,
        alternatives: str,
        ipa: str,
        audio_data: bytes | None,
        audio_filename: str | None,
        source: str | None,
    ) -> int:
        return await asyncio.to_thread(
            self._write_sync,
            username=username,
            word=word,
            lemma=lemma,
            sentence=sentence,
            translation=translation,
            alternatives=alternatives,
            ipa=ipa,
            audio_data=audio_data,
            audio_filename=audio_filename,
            source=source,
        )

    def _write_sync(
        self,
        *,
        username: str,
        word: str,
        lemma: str,
        sentence: str | None,
        translation: str,
        alternatives: str,
        ipa: str,
        audio_data: bytes | None,
        audio_filename: str | None,
        source: str | None,
    ) -> int:
        col_path = self.collection_path(username)
        col_path.parent.mkdir(parents=True, exist_ok=True)

        if audio_data is not None and audio_filename:
            media = self.media_dir(username)
            media.mkdir(parents=True, exist_ok=True)
            (media / audio_filename).write_bytes(audio_data)

        col = Collection(str(col_path))
        try:
            model = _ensure_notetype(col)
            deck_id = col.decks.id(self._deck_name)
            assert deck_id is not None

            note = col.new_note(model)
            note["Word"] = word
            note["Lemma"] = lemma
            note["Sentence"] = sentence or ""
            note["Translation"] = translation
            note["Alternatives"] = alternatives
            note["IPA"] = ipa
            note["Audio"] = f"[sound:{audio_filename}]" if audio_filename else ""
            note["Source"] = source or ""
            note["DateAdded"] = datetime.now(UTC).date().isoformat()

            col.add_note(note, deck_id=deck_id)

            card_ids = col.find_cards(f"nid:{note.id}")
            return int(card_ids[0])
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
