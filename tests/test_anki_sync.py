from pathlib import Path
from typing import Any

import pytest
from anki.collection import Collection
from anki.sync_pb2 import SyncAuth, SyncCollectionResponse

from vocab_api.anki_sync import AnkiSyncWriter, parse_credentials_json
from vocab_api.anki_writer import VocabCardContent


def _content(**overrides: Any) -> VocabCardContent:
    base: dict[str, Any] = {
        "word": "expedition",
        "lemma": "expedition",
        "sentence": "A grand expedition north.",
        "translation": "die Expedition",
        "alternatives": "die Reise",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "audio_data": b"FAKE-MP3",
        "audio_filename": "abc123.mp3",
        "source": "test-source",
    }
    base.update(overrides)
    return VocabCardContent(**base)


def _patch_sync(monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
    """Stub Collection.sync_login and Collection.sync_collection.

    Returns (login_calls, sync_calls) lists that the tests can assert on."""
    login_calls: list[tuple[str, str, str]] = []
    sync_calls: list[tuple[str, bool]] = []

    def fake_login(self: Collection, username: str, password: str, endpoint: str) -> SyncAuth:
        login_calls.append((username, password, endpoint))
        return SyncAuth(hkey=f"hkey-{username}", endpoint=endpoint)

    def fake_sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncCollectionResponse:
        sync_calls.append((auth.hkey, sync_media))
        return SyncCollectionResponse()

    monkeypatch.setattr(Collection, "sync_login", fake_login)
    monkeypatch.setattr(Collection, "sync_collection", fake_sync)
    return login_calls, sync_calls


async def test_writes_card_into_shadow_and_pushes_to_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    login_calls, sync_calls = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test:27701",
        credentials={"alice": "pw1"},
    )

    card_id = await writer.write_card(username="alice", content=_content())

    assert card_id > 0
    assert login_calls == [("alice", "pw1", "http://anki-sync.test:27701")]
    # Sync down (no media) before mutating, then sync up (with media).
    assert sync_calls == [("hkey-alice", False), ("hkey-alice", True)]

    shadow = tmp_path / "alice" / "collection.anki2"
    assert shadow.exists()
    col = Collection(str(shadow))
    try:
        note = col.get_card(card_id).note()
        assert note["Word"] == "expedition"
        assert note["Translation"] == "die Expedition"
        assert note["Audio"] == "[sound:abc123.mp3]"
    finally:
        col.close()

    media_file = tmp_path / "alice" / "collection.media" / "abc123.mp3"
    assert media_file.read_bytes() == b"FAKE-MP3"


async def test_caches_auth_per_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    login_calls, _sync_calls = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )

    await writer.write_card(username="alice", content=_content(word="first", lemma="first"))
    await writer.write_card(username="alice", content=_content(word="second", lemma="second"))

    # sync_login should be called only once; the second write reuses the hkey.
    assert login_calls == [("alice", "pw1", "http://anki-sync.test")]


async def test_isolates_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    login_calls, _sync_calls = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw-alice", "bob": "pw-bob"},
    )

    await writer.write_card(username="alice", content=_content())
    await writer.write_card(username="bob", content=_content(word="z", lemma="z"))

    assert (tmp_path / "alice" / "collection.anki2").exists()
    assert (tmp_path / "bob" / "collection.anki2").exists()
    assert ("alice", "pw-alice", "http://anki-sync.test") in login_calls
    assert ("bob", "pw-bob", "http://anki-sync.test") in login_calls


async def test_missing_credentials_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={},
    )

    with pytest.raises(RuntimeError, match="anki-sync credentials.*alice"):
        await writer.write_card(username="alice", content=_content())


def test_parse_credentials_json_handles_empty():
    assert parse_credentials_json("") == {}
    assert parse_credentials_json("{}") == {}


def test_parse_credentials_json_returns_username_password_map():
    creds = parse_credentials_json('{"alice": "pw1", "bob": "pw2"}')
    assert creds == {"alice": "pw1", "bob": "pw2"}


def test_parse_credentials_json_rejects_non_string_values():
    with pytest.raises(ValueError, match="must be a string"):
        parse_credentials_json('{"alice": 123}')
