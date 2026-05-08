from pathlib import Path

import pytest
from anki.collection import Collection

from vocab_api.anki_writer import VOCAB_FIELDS, VOCAB_NOTETYPE, AnkiWriter


def _open_collection(root: Path, username: str) -> Collection:
    return Collection(str(root / username / "collection.anki2"))


async def test_write_card_creates_collection_card_and_media(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    card_id = await writer.write_card(
        username="alice",
        word="expedition",
        lemma="expedition",
        sentence="A grand expedition north.",
        translation="die Expedition",
        alternatives="die Reise",
        ipa="/ˌɛkspɪˈdɪʃən/",
        audio_data=b"FAKE-MP3",
        audio_filename="abc123.mp3",
        source="test-source",
    )

    assert card_id > 0
    assert (tmp_path / "alice" / "collection.anki2").exists()
    assert (tmp_path / "alice" / "collection.media" / "abc123.mp3").read_bytes() == b"FAKE-MP3"


async def test_vocab_notetype_has_expected_fields(tmp_path: Path):
    writer = AnkiWriter(root=tmp_path)
    await writer.write_card(
        username="alice",
        word="x",
        lemma="x",
        sentence=None,
        translation="y",
        alternatives="",
        ipa="",
        audio_data=None,
        audio_filename=None,
        source=None,
    )

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
            username="alice",
            word=word,
            lemma=word,
            sentence=None,
            translation="x",
            alternatives="",
            ipa="",
            audio_data=None,
            audio_filename=None,
            source=None,
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
        word="expedition",
        lemma="expedition",
        sentence="A grand expedition.",
        translation="die Expedition",
        alternatives="die Reise, der Forschungsausflug",
        ipa="/ˌɛkspɪˈdɪʃən/",
        audio_data=b"x",
        audio_filename="a.mp3",
        source="https://example.com/article",
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
    await writer.write_card(
        username="alice",
        word="x",
        lemma="x",
        sentence=None,
        translation="y",
        alternatives="",
        ipa="",
        audio_data=None,
        audio_filename=None,
        source=None,
    )
    await writer.write_card(
        username="bob",
        word="z",
        lemma="z",
        sentence=None,
        translation="w",
        alternatives="",
        ipa="",
        audio_data=None,
        audio_filename=None,
        source=None,
    )

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
        username="alice",
        word="x",
        lemma="x",
        sentence=None,
        translation="y",
        alternatives="",
        ipa="",
        audio_data=None,
        audio_filename=audio_filename,
        source=None,
    )

    col = _open_collection(tmp_path, "alice")
    try:
        note = col.get_card(card_id).note()
        assert note["Audio"] == ""
    finally:
        col.close()
