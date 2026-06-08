from pathlib import Path

import pytest
from anki.collection import Collection

from vocab_api.anki_writer import (
    VOCAB_FIELDS,
    VOCAB_NOTETYPE,
    AnkiWriter,
    VocabCardContent,
    deck_name_for,
    update_vocab_note,
)


def _open_collection(root: Path, username: str) -> Collection:
    return Collection(str(root / username / "collection.anki2"))


def _content(**overrides: object) -> VocabCardContent:
    base: dict[str, object] = {
        "word": "x",
        "lemma": "x",
        "sentence": None,
        "cloze_sentence": "",
        "translation": "y",
        "alternatives": "",
        "ipa": "",
        "sense_label": "",
        "collocations": "",
        "extra_examples": "",
        "audio_data": None,
        "audio_filename": None,
        "source": None,
        "alt_lemma": "",
        "alt_reason": "",
        "alt_translation": "",
        "alt_ipa": "",
        "alt_examples": "",
        "alt_audio_data": None,
        "alt_audio_filename": None,
    }
    base.update(overrides)
    return VocabCardContent(**base)  # type: ignore[arg-type]


async def test_write_card_creates_collection_card_and_media(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="expedition",
            lemma="expedition",
            sentence="A grand expedition north.",
            translation="die Expedition",
            alternatives="die Reise",
            ipa="/ˌɛkspɪˈdɪʃən/",
            audio_data=b"FAKE-MP3",
            audio_filename="abc123.mp3",
            source="test-source",
        ),
    )

    assert card_id > 0
    assert (tmp_path / "alice" / "collection.anki2").exists()
    assert (tmp_path / "alice" / "collection.media" / "abc123.mp3").read_bytes() == b"FAKE-MP3"


async def test_vocab_notetype_has_expected_fields(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        m = col.models.by_name(VOCAB_NOTETYPE)
        assert m is not None
        field_names = [f["name"] for f in m["flds"]]
        assert field_names == VOCAB_FIELDS
    finally:
        col.close()


async def test_second_write_reuses_notetype(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    for word in ("first", "second"):
        await writer.write_card(
            username="alice", content=_content(word=word, lemma=word, translation="x")
        )

    col = _open_collection(tmp_path, "alice")
    try:
        vocab_models = [m for m in col.models.all() if m["name"] == VOCAB_NOTETYPE]
        assert len(vocab_models) == 1
        assert col.card_count() == 2
    finally:
        col.close()


async def test_card_fields_contain_provided_values(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="expedition",
            lemma="expedition",
            sentence="A grand expedition.",
            translation="die Expedition",
            alternatives="die Reise, der Forschungsausflug",
            ipa="/ˌɛkspɪˈdɪʃən/",
            audio_data=b"x",
            audio_filename="a.mp3",
            source="https://example.com/article",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        card = col.get_card(card_id)
        note = card.note()
        assert note["Word"] == "expedition"
        assert note["Lemma"] == "expedition"
        assert note["Sentence"] == "A grand expedition."
        assert note["Translation"] == "die Expedition"
        assert note["Alternatives"] == "die Reise, der Forschungsausflug"
        assert note["IPA"] == "/ˌɛkspɪˈdɪʃən/"
        assert note["Audio"] == "[sound:a.mp3]"
        assert note["Source"] == "https://example.com/article"
        assert note["DateAdded"] != ""
    finally:
        col.close()


async def test_write_card_isolates_users(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())
    await writer.write_card(username="bob", content=_content(word="z", lemma="z", translation="w"))

    alice = _open_collection(tmp_path, "alice")
    bob = _open_collection(tmp_path, "bob")
    try:
        assert alice.card_count() == 1
        assert bob.card_count() == 1
    finally:
        alice.close()
        bob.close()


@pytest.mark.parametrize("audio_filename", [None, ""])
async def test_audio_field_empty_when_no_audio(tmp_path: Path, audio_filename: str | None):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice", content=_content(audio_filename=audio_filename)
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["Audio"] == ""
    finally:
        col.close()


async def test_cloze_sentence_is_persisted_to_note(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="train",
            lemma="train",
            sentence="The train leaves at 8.",
            cloze_sentence="The ___ leaves at 8.",
            translation="der Zug",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["ClozeSentence"] == "The ___ leaves at 8."
        assert note["Sentence"] == "The train leaves at 8."
    finally:
        col.close()


async def test_notetype_includes_cloze_sentence_field(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        field_names = [f["name"] for f in model["flds"]]
        assert "ClozeSentence" in field_names
    finally:
        col.close()


async def test_front_template_uses_cloze_sentence_and_translation_hint(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        qfmt = model["tmpls"][0]["qfmt"]
        # Front must show the gap sentence and the German translation as hint —
        # the bare {{Word}} must not appear on the front, otherwise the card
        # gives away the answer (active-recall principle, see #23).
        assert "{{ClozeSentence}}" in qfmt
        assert "{{Translation}}" in qfmt
        assert "{{Word}}" not in qfmt
    finally:
        col.close()


async def test_back_template_shows_lemma_audio_and_full_sentence(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        afmt = model["tmpls"][0]["afmt"]
        # Headword is the dictionary form, not the user's inflected input —
        # the learner studies the lemma; the surface form still shows up in
        # the example sentence and the cloze gap.
        assert "{{Lemma}}" in afmt
        assert "{{Word}}" not in afmt
        assert "{{Audio}}" in afmt
        assert "{{Sentence}}" in afmt
    finally:
        col.close()


async def test_de_en_back_renders_audio_in_its_own_block(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        afmt = model["tmpls"][0]["afmt"]
        # Audio gets its own paragraph instead of sitting inline next to the
        # small IPA text — bigger tap target on mobile, and consistent with
        # the EN→DE back where audio already stands on its own line.
        assert "{{#Audio}}<p>{{Audio}}</p>{{/Audio}}" in afmt
        assert "{{/IPA}}{{Audio}}" not in afmt
    finally:
        col.close()


async def test_extra_examples_persist_to_note_and_render_on_back(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="take effect",
            lemma="take effect",
            translation="in Kraft treten",
            extra_examples=(
                "The law takes effect on Jan 1st.<br>When does the change take effect?"
            ),
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["ExtraExamples"] == (
            "The law takes effect on Jan 1st.<br>When does the change take effect?"
        )
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        afmt = model["tmpls"][0]["afmt"]
        assert "{{#ExtraExamples}}" in afmt and "{{/ExtraExamples}}" in afmt
        assert "{{ExtraExamples}}" in afmt
    finally:
        col.close()


async def test_collocations_persist_to_note_and_render_on_back(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="decision",
            lemma="decision",
            translation="die Entscheidung",
            collocations="make a decision · tough decision · reach a decision",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["Collocations"] == ("make a decision · tough decision · reach a decision")
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        afmt = model["tmpls"][0]["afmt"]
        # Anki's conditional rendering hides the block for monosemous words
        # so the back has no empty line where collocations would be.
        assert "{{#Collocations}}" in afmt and "{{/Collocations}}" in afmt
        assert "{{Collocations}}" in afmt
    finally:
        col.close()


async def test_sense_label_is_persisted_to_note(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="train",
            lemma="train",
            translation="der Zug",
            sense_label="Eisenbahn",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["SenseLabel"] == "Eisenbahn"
    finally:
        col.close()


async def test_front_template_renders_sense_label_when_present(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        qfmt = model["tmpls"][0]["qfmt"]
        # Anki's conditional field rendering: {{#SenseLabel}}…{{/SenseLabel}}
        # shows the inner block only when SenseLabel is non-empty. Otherwise
        # the hint just reads "(die Bank)" without a stray comma.
        assert "{{#SenseLabel}}" in qfmt and "{{/SenseLabel}}" in qfmt
        assert "{{SenseLabel}}" in qfmt
    finally:
        col.close()


def _card_deck_name(col: Collection, card_id: int) -> str:
    card = col.get_card(card_id)
    name = col.decks.name(card.did)
    assert isinstance(name, str)
    return name


async def test_de_en_card_lands_in_lang_subdeck(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(username="alice", content=_content(), lang="en")
    col = _open_collection(tmp_path, "alice")
    try:
        assert _card_deck_name(col, card_id) == "Englisch::DE→EN"
    finally:
        col.close()


async def test_en_de_card_lands_in_recognition_subdeck(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice", content=_content(), direction="en_de", lang="en"
    )
    col = _open_collection(tmp_path, "alice")
    try:
        assert _card_deck_name(col, card_id) == "Englisch::EN→DE"
    finally:
        col.close()


async def test_both_direction_routes_each_card_to_its_own_subdeck(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    # write_card returns the first card; we look up both cards on the note.
    await writer.write_card(username="alice", content=_content(), direction="both", lang="en")
    col = _open_collection(tmp_path, "alice")
    try:
        cards = list(col.find_cards(""))
        assert len(cards) == 2
        deck_names = sorted(_card_deck_name(col, cid) for cid in cards)
        assert deck_names == ["Englisch::DE→EN", "Englisch::EN→DE"]
    finally:
        col.close()


async def test_unknown_lang_creates_uppercase_fallback_subdeck(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(username="alice", content=_content(), lang="tlh")
    col = _open_collection(tmp_path, "alice")
    try:
        assert _card_deck_name(col, card_id) == "TLH::DE→EN"
    finally:
        col.close()


async def test_existing_card_keeps_its_deck_when_lang_switches(tmp_path: Path):
    # First note is English; second note is Spanish. The existing English
    # card must stay in its deck — the per-template `did` only influences
    # newly generated cards (#34).
    writer = AnkiWriter(root=tmp_path)
    en_card = await writer.write_card(
        username="alice", content=_content(word="train", lemma="train"), lang="en"
    )
    es_card = await writer.write_card(
        username="alice", content=_content(word="tren", lemma="tren"), lang="es"
    )

    col = _open_collection(tmp_path, "alice")
    try:
        assert _card_deck_name(col, en_card) == "Englisch::DE→EN"
        assert _card_deck_name(col, es_card) == "Spanisch::DE→EN"
    finally:
        col.close()


def test_deck_name_for_known_language_uses_german_display_name():
    assert deck_name_for(lang="en", template_name="DE→EN") == "Englisch::DE→EN"
    assert deck_name_for(lang="es", template_name="EN→DE") == "Spanisch::EN→DE"
    assert deck_name_for(lang="nl", template_name="DE→EN") == "Niederländisch::DE→EN"


def test_deck_name_for_unknown_language_falls_back_to_uppercase_code():
    # No mapping for "tlh" — keep emitting a sensible deck rather than crash.
    assert deck_name_for(lang="tlh", template_name="DE→EN") == "TLH::DE→EN"


async def test_default_direction_creates_only_de_en_template(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        names = [t["name"] for t in model["tmpls"]]
        assert names == ["DE→EN"]
        # Sanity: this single template is the production-direction one
        # (cloze gap on the front, word on the back).
        assert "{{ClozeSentence}}" in model["tmpls"][0]["qfmt"]
    finally:
        col.close()


async def test_both_direction_creates_two_templates(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content(), direction="both")

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        names = [t["name"] for t in model["tmpls"]]
        assert names == ["DE→EN", "EN→DE"]
        # EN→DE template shows the bare lemma on the front.
        en_de = model["tmpls"][1]
        assert en_de["qfmt"].strip() == "<h2>{{Lemma}}</h2>{{#IPA}}<small>{{IPA}}</small>{{/IPA}}"
        assert "{{Translation}}" in en_de["afmt"]
    finally:
        col.close()


async def test_en_de_only_direction_creates_only_recognition_template(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content(), direction="en_de")

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        names = [t["name"] for t in model["tmpls"]]
        assert names == ["EN→DE"]
    finally:
        col.close()


async def test_upgrading_from_de_en_to_both_adds_recognition_template(tmp_path: Path):
    # Alice's collection starts in single-direction mode (today's default).
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(username="alice", content=_content())
    # She flips to 'both' later. The next write must add the EN→DE template
    # idempotently without disturbing the existing DE→EN one.
    await writer.write_card(
        username="alice", content=_content(word="second", lemma="second"), direction="both"
    )

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        names = [t["name"] for t in model["tmpls"]]
        assert names == ["DE→EN", "EN→DE"]
    finally:
        col.close()


async def test_legacy_card1_template_is_renamed_to_de_en(tmp_path: Path):
    # Simulate a pre-#25 collection whose first template is still called
    # "Card 1". The migration must rename it to "DE→EN" so behavior stays
    # consistent across freshly-created and legacy collections.
    writer = AnkiWriter(root=tmp_path)
    col_path = writer.collection_path("alice")
    col_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = Collection(str(col_path))
    try:
        model = legacy.models.new(VOCAB_NOTETYPE)
        for field_name in VOCAB_FIELDS:
            legacy.models.add_field(model, legacy.models.new_field(field_name))
        template = legacy.models.new_template("Card 1")
        template["qfmt"] = "legacy {{Word}}"
        template["afmt"] = "legacy {{Translation}}"
        legacy.models.add_template(model, template)
        legacy.models.add(model)
    finally:
        legacy.close()

    await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        names = [t["name"] for t in model["tmpls"]]
        assert names == ["DE→EN"]
    finally:
        col.close()


async def test_alt_fields_persist_to_note_and_render_conditionally(tmp_path: Path):
    # When a more idiomatic alternative exists (#60), the card carries an
    # extra block: marker line ("dated — more common: …"), alt IPA + translation,
    # alt examples and an alt audio button. The whole block is wrapped in
    # `{{#AltLemma}}…{{/AltLemma}}` so cards without an alt look unchanged.
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="weary",
            lemma="weary",
            translation="müde",
            alt_lemma="exhausted",
            alt_reason="dated",
            alt_translation="erschöpft",
            alt_ipa="/ɪɡˈzɔːstɪd/",
            alt_examples=(
                "She was exhausted after the hike.<br>I'm too exhausted to cook tonight."
            ),
            alt_audio_data=b"FAKE-MP3-ALT",
            alt_audio_filename="alt123.mp3",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["AltLemma"] == "exhausted"
        assert note["AltReason"] == "dated"
        assert note["AltTranslation"] == "erschöpft"
        assert note["AltIPA"] == "/ɪɡˈzɔːstɪd/"
        assert note["AltExamples"] == (
            "She was exhausted after the hike.<br>I'm too exhausted to cook tonight."
        )
        assert note["AltAudio"] == "[sound:alt123.mp3]"

        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        afmt = model["tmpls"][0]["afmt"]
        # Conditional block: hidden when AltLemma empty (the common case).
        assert "{{#AltLemma}}" in afmt and "{{/AltLemma}}" in afmt
        assert "{{AltLemma}}" in afmt
        assert "{{AltTranslation}}" in afmt
        assert "{{AltReason}}" in afmt
        assert "{{AltAudio}}" in afmt
    finally:
        col.close()

    # Alt-audio media file lands alongside the primary audio.
    assert (tmp_path / "alice" / "collection.media" / "alt123.mp3").read_bytes() == b"FAKE-MP3-ALT"


async def test_card_without_alt_lemma_has_empty_alt_fields(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(username="alice", content=_content())

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["AltLemma"] == ""
        assert note["AltAudio"] == ""
    finally:
        col.close()


async def test_existing_notetype_gets_cloze_sentence_field_added(tmp_path: Path):
    # Simulate an Anki collection that already has the legacy 9-field "Vocab"
    # notetype from before #23 landed. _ensure_notetype must idempotently add
    # the new ClozeSentence field instead of failing silently.
    writer = AnkiWriter(root=tmp_path)
    col_path = writer.collection_path("alice")
    col_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = Collection(str(col_path))
    try:
        model = legacy.models.new(VOCAB_NOTETYPE)
        for field_name in (
            "Word",
            "Lemma",
            "Sentence",
            "Translation",
            "Alternatives",
            "IPA",
            "Audio",
            "Source",
            "DateAdded",
        ):
            legacy.models.add_field(model, legacy.models.new_field(field_name))
        template = legacy.models.new_template("Card 1")
        # Anki refuses to save a template with no field placeholders — give it
        # a valid (but visibly old) shape so we can prove the migration runs.
        template["qfmt"] = "legacy {{Word}}"
        template["afmt"] = "legacy {{Translation}}"
        legacy.models.add_template(model, template)
        legacy.models.add(model)
    finally:
        legacy.close()

    await writer.write_card(
        username="alice",
        content=_content(cloze_sentence="A ___ test."),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        model = col.models.by_name(VOCAB_NOTETYPE)
        assert model is not None
        field_names = [f["name"] for f in model["flds"]]
        # New field appended (Anki forbids re-ordering existing fields with
        # data without a media migration; we settle for set-equality here).
        assert set(field_names) == set(VOCAB_FIELDS)
        # Legacy template strings must be refreshed so old cards render with
        # the new layout after the migration runs.
        qfmt = model["tmpls"][0]["qfmt"]
        assert "legacy" not in qfmt
        assert "{{ClozeSentence}}" in qfmt
    finally:
        col.close()


async def test_update_vocab_note_changes_cloze_sentence_field(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="train",
            lemma="train",
            cloze_sentence="The ___ was late.",
            translation="der Zug",
        ),
    )

    col = _open_collection(tmp_path, "alice")
    try:
        update_vocab_note(col, card_id, "A ___ arrived early.")
        note = col.get_card(card_id).note()
        assert note["ClozeSentence"] == "A ___ arrived early."
    finally:
        col.close()


async def test_update_card_persists_changed_cloze_sentence(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        content=_content(
            word="train",
            lemma="train",
            cloze_sentence="The ___ was late.",
            translation="der Zug",
        ),
    )

    await writer.update_card(
        username="alice",
        card_id=card_id,
        cloze_sentence="A ___ arrived early.",
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["ClozeSentence"] == "A ___ arrived early."
    finally:
        col.close()
