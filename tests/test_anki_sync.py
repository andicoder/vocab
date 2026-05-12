from collections.abc import Iterable
from dataclasses import dataclass, field
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
        "cloze_sentence": "A grand ___ north.",
        "translation": "die Expedition",
        "alternatives": "die Reise",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sense_label": "",
        "collocations": "",
        "extra_examples": "",
        "audio_data": b"FAKE-MP3",
        "audio_filename": "abc123.mp3",
        "source": "test-source",
    }
    base.update(overrides)
    return VocabCardContent(**base)


@dataclass
class SyncHarness:
    """Captures every call AnkiSyncWriter makes against `Collection`.

    Tests can hand a list of `SyncCollectionResponse` values via
    `sync_responses` to drive specific server states (NO_CHANGES,
    FULL_SYNC, etc.). When the list is exhausted — or when no list is
    given — sync_collection returns a default `NO_CHANGES` response,
    matching the happy path.

    Exposing all three call lists (login, sync, full) on one object lets
    a single test assert on the full interaction pattern instead of
    chasing tuples through the test body."""

    login_calls: list[tuple[str, str, str]] = field(default_factory=list)
    sync_calls: list[tuple[str, bool]] = field(default_factory=list)
    full_calls: list[tuple[str, int, bool]] = field(default_factory=list)


def _patch_sync(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sync_responses: Iterable[SyncCollectionResponse] | None = None,
) -> SyncHarness:
    harness = SyncHarness()
    responses = iter(sync_responses) if sync_responses is not None else None

    def fake_login(self: Collection, username: str, password: str, endpoint: str) -> SyncAuth:
        harness.login_calls.append((username, password, endpoint))
        return SyncAuth(hkey=f"hkey-{username}", endpoint=endpoint)

    def fake_sync(self: Collection, auth: SyncAuth, sync_media: bool) -> SyncCollectionResponse:
        harness.sync_calls.append((auth.hkey, sync_media))
        if responses is None:
            return SyncCollectionResponse()
        return next(responses)

    def fake_full(
        self: Collection, *, auth: SyncAuth, server_usn: int | None, upload: bool
    ) -> None:
        harness.full_calls.append((auth.hkey, server_usn or -1, upload))

    monkeypatch.setattr(Collection, "sync_login", fake_login)
    monkeypatch.setattr(Collection, "sync_collection", fake_sync)
    monkeypatch.setattr(Collection, "full_upload_or_download", fake_full)
    return harness


async def test_writes_card_into_shadow_and_pushes_to_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test:27701",
        credentials={"alice": "pw1"},
    )

    card_id = await writer.write_card(username="alice", content=_content())

    assert card_id > 0
    assert harness.login_calls == [("alice", "pw1", "http://anki-sync.test:27701")]
    # Sync down (no media) before mutating, then sync up (with media).
    assert harness.sync_calls == [("hkey-alice", False), ("hkey-alice", True)]

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
    harness = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )

    await writer.write_card(username="alice", content=_content(word="first", lemma="first"))
    await writer.write_card(username="alice", content=_content(word="second", lemma="second"))

    # sync_login should be called only once; the second write reuses the hkey.
    assert harness.login_calls == [("alice", "pw1", "http://anki-sync.test")]


async def test_isolates_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    harness = _patch_sync(monkeypatch)
    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw-alice", "bob": "pw-bob"},
    )

    await writer.write_card(username="alice", content=_content())
    await writer.write_card(username="bob", content=_content(word="z", lemma="z"))

    assert (tmp_path / "alice" / "collection.anki2").exists()
    assert (tmp_path / "bob" / "collection.anki2").exists()
    assert ("alice", "pw-alice", "http://anki-sync.test") in harness.login_calls
    assert ("bob", "pw-bob", "http://anki-sync.test") in harness.login_calls


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


_INCREMENTAL_REQUIRED = [
    SyncCollectionResponse.NO_CHANGES,
    SyncCollectionResponse.NORMAL_SYNC,
]
_FULL_REQUIRED = [
    SyncCollectionResponse.FULL_SYNC,
    SyncCollectionResponse.FULL_DOWNLOAD,
    SyncCollectionResponse.FULL_UPLOAD,
]


@pytest.mark.parametrize("pull_required", _INCREMENTAL_REQUIRED)
async def test_pull_without_full_sync_signal_does_not_reattach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pull_required: int
):
    # Happy paths — `sync_collection` returns NO_CHANGES or NORMAL_SYNC
    # for both pull and push, so the writer never touches
    # `full_upload_or_download`.
    pull = SyncCollectionResponse(required=pull_required)
    push = SyncCollectionResponse(required=SyncCollectionResponse.NO_CHANGES)
    harness = _patch_sync(monkeypatch, sync_responses=[pull, push])

    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )
    await writer.write_card(username="alice", content=_content())

    assert harness.full_calls == []


@pytest.mark.parametrize("pull_required", _FULL_REQUIRED)
async def test_pull_with_full_sync_signal_downloads_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pull_required: int
):
    # Regression for #40: any of the three "full sync needed" signals
    # detaches the shadow from the server. Writer must reattach by
    # downloading the server's canonical state — uploading our fresh
    # shadow would wipe the user's real cards.
    pull = SyncCollectionResponse(required=pull_required, server_media_usn=42)
    push = SyncCollectionResponse(required=SyncCollectionResponse.NO_CHANGES)
    harness = _patch_sync(monkeypatch, sync_responses=[pull, push])

    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )
    card_id = await writer.write_card(username="alice", content=_content())

    assert card_id > 0
    assert harness.full_calls == [("hkey-alice", 42, False)]


async def test_push_with_full_upload_signal_uploads_local_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # On the first write to a fresh per-user collection the server is
    # empty and our shadow has one new note; the server signals
    # FULL_UPLOAD, which is the legitimate "you have more, push it as a
    # full sync" path. Uploading is the *only* way the note actually
    # reaches the server in that scenario.
    pull = SyncCollectionResponse(required=SyncCollectionResponse.NO_CHANGES)
    push = SyncCollectionResponse(required=SyncCollectionResponse.FULL_UPLOAD, server_media_usn=7)
    harness = _patch_sync(monkeypatch, sync_responses=[pull, push])

    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )
    await writer.write_card(username="alice", content=_content())

    assert harness.full_calls == [("hkey-alice", 7, True)]


@pytest.mark.parametrize(
    "push_required",
    [SyncCollectionResponse.FULL_SYNC, SyncCollectionResponse.FULL_DOWNLOAD],
)
async def test_push_with_full_download_or_full_sync_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, push_required: int
):
    # FULL_DOWNLOAD or FULL_SYNC after our write means the server claims
    # canonical data that we don't have — accepting either would erase
    # the just-written note. Fail loud so the entry stays unsynced and
    # the next worker tick retries.
    pull = SyncCollectionResponse(required=SyncCollectionResponse.NO_CHANGES)
    push = SyncCollectionResponse(required=push_required, server_media_usn=1)
    harness = _patch_sync(monkeypatch, sync_responses=[pull, push])

    writer = AnkiSyncWriter(
        shadow_root=tmp_path,
        sync_endpoint="http://anki-sync.test",
        credentials={"alice": "pw1"},
    )
    with pytest.raises(RuntimeError, match="full sync required after write"):
        await writer.write_card(username="alice", content=_content())
    assert harness.full_calls == []


def test_parse_credentials_json_handles_empty():
    assert parse_credentials_json("") == {}
    assert parse_credentials_json("{}") == {}


def test_parse_credentials_json_returns_username_password_map():
    creds = parse_credentials_json('{"alice": "pw1", "bob": "pw2"}')
    assert creds == {"alice": "pw1", "bob": "pw2"}


def test_parse_credentials_json_rejects_non_string_values():
    with pytest.raises(ValueError, match="must be a string"):
        parse_credentials_json('{"alice": 123}')
