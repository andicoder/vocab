"""End-to-end sync against a real anki-sync-server.

The anki Python package ships its own sync server as `python -m
anki.syncserver`. Using that as the test backend gives us the strongest
guarantee we can get short of pointing at AnkiWeb itself: the *same*
Rust backend serializes the wire format on both sides, so any
client/server mismatch shows up here.

Why this exists at all: the unit tests in `test_anki_sync.py` mock
`Collection.sync_collection`, which means they can't catch a bug like
the one in #40 — the writer was ignoring `SyncOutput.required` and the
mock kept returning the happy default. An end-to-end run against the
real server caught that immediately because the protocol genuinely
demands a full sync on a fresh shadow."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from anki.collection import Collection
from anki.sync_pb2 import SyncAuth, SyncCollectionResponse

from vocab_api.anki_sync import AnkiSyncWriter
from vocab_api.anki_writer import NoteFieldUpdate, VocabCardContent


class _SyncServer(NamedTuple):
    endpoint: str
    username: str
    password: str
    # Its own collection, for tests that can't share one with `username`.
    legacy_username: str
    legacy_password: str


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.1)
    raise TimeoutError(f"anki-sync-server not listening on {host}:{port}: {last_err}")


@pytest.fixture(scope="session")
def anki_sync_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_SyncServer]:
    server_root = tmp_path_factory.mktemp("anki-sync-server")
    port = _find_free_port()
    username = "ci"
    password = "cipw"
    legacy_username = "legacy"
    legacy_password = "legacypw"
    proc = subprocess.Popen(
        [sys.executable, "-m", "anki.syncserver"],
        env={
            **os.environ,
            "SYNC_USER1": f"{username}:{password}",
            "SYNC_USER2": f"{legacy_username}:{legacy_password}",
            "SYNC_HOST": "127.0.0.1",
            "SYNC_PORT": str(port),
            "SYNC_BASE": str(server_root),
            # Quiet by default; flip to "anki=debug" when chasing a sync bug.
            "RUST_LOG": os.environ.get("RUST_LOG", "anki=warn"),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port("127.0.0.1", port, timeout_s=10.0)
        yield _SyncServer(
            endpoint=f"http://127.0.0.1:{port}/",
            username=username,
            password=password,
            legacy_username=legacy_username,
            legacy_password=legacy_password,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _content(**overrides: object) -> VocabCardContent:
    base: dict[str, object] = {
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
    return VocabCardContent(**base)  # type: ignore[arg-type]


def _sync_to_steady_state(col: Collection, *, username: str, password: str, endpoint: str) -> None:
    """Connect a fresh Collection to the server and resolve any FULL_SYNC.

    Mirrors the production-side reattach logic (#40) so the verification
    client doesn't reproduce the very bug it's supposed to detect."""
    auth = col.sync_login(username, password, endpoint)
    out = col.sync_collection(auth, sync_media=False)
    if out.required not in (
        SyncCollectionResponse.NO_CHANGES,
        SyncCollectionResponse.NORMAL_SYNC,
    ):
        col.full_upload_or_download(auth=auth, server_usn=out.server_media_usn, upload=False)
        col.sync_collection(auth, sync_media=False)


async def test_written_card_reaches_anki_sync_server(
    anki_sync_server: _SyncServer, tmp_path: Path
) -> None:
    server = anki_sync_server

    writer = AnkiSyncWriter(
        shadow_root=tmp_path / "shadow",
        sync_endpoint=server.endpoint,
        credentials={server.username: server.password},
    )
    await writer.write_card(
        username=server.username,
        content=_content(word="expedition", lemma="expedition"),
    )

    # Independent verification path: open a fresh client collection with
    # nothing in common with the writer's shadow, sync it down from the
    # server, and confirm the card is actually there. If anything in the
    # write→sync chain silently dropped the card (#40 was exactly that),
    # this client wouldn't see it.
    verify_path = tmp_path / "verify" / "collection.anki2"
    verify_path.parent.mkdir(parents=True)
    col = Collection(str(verify_path))
    try:
        _sync_to_steady_state(
            col,
            username=server.username,
            password=server.password,
            endpoint=server.endpoint,
        )
        assert col.card_count() == 1
        card_ids = col.find_cards("")
        note = col.get_card(card_ids[0]).note()
        assert note["Word"] == "expedition"
        assert note["Translation"] == "die Expedition"
    finally:
        col.close()


def _push_pre_exercise_notetype(col: Collection, *, auth: SyncAuth) -> None:
    """Put the server on the notetype it had before the Übung card existed."""
    model = col.models.by_name("Vocab")
    assert model is not None
    model["tmpls"] = [t for t in model["tmpls"] if t["name"] != "Übung"]
    col.models.update_dict(model)
    model = col.models.by_name("Vocab")
    assert model is not None
    for field in [f for f in model["flds"] if f["name"].startswith("Exercise")]:
        col.models.remove_field(model, field)
    col.models.update_dict(model)
    out = col.sync_collection(auth, sync_media=False)
    assert out.required == SyncCollectionResponse.FULL_SYNC
    col.full_upload_or_download(auth=auth, server_usn=out.server_media_usn, upload=True)


async def test_update_reaches_a_server_still_on_the_pre_exercise_notetype(
    anki_sync_server: _SyncServer, tmp_path: Path
) -> None:
    # Production before this fix: the first Anki write after the Übung card
    # deploy added its fields, the server answered FULL_SYNC, and the writer
    # refused every night. Mocks can't catch this, only the real protocol does.
    server = anki_sync_server
    username, password = server.legacy_username, server.legacy_password
    credentials = {username: password}

    seed_writer = AnkiSyncWriter(
        shadow_root=tmp_path / "seed",
        sync_endpoint=server.endpoint,
        credentials=credentials,
    )
    card_id = await seed_writer.write_card(username=username, content=_content())

    phone_path = tmp_path / "phone" / "collection.anki2"
    phone_path.parent.mkdir(parents=True)
    phone = Collection(str(phone_path))
    try:
        _sync_to_steady_state(phone, username=username, password=password, endpoint=server.endpoint)
        auth = phone.sync_login(username, password, server.endpoint)
        _push_pre_exercise_notetype(phone, auth=auth)
    finally:
        phone.close()

    writer = AnkiSyncWriter(
        shadow_root=tmp_path / "shadow",
        sync_endpoint=server.endpoint,
        credentials=credentials,
    )
    await writer.update_card(
        username=username,
        card_id=card_id,
        fields=NoteFieldUpdate(cloze_sentence="A ___ north.", exercise_front="<p>Frage</p>"),
    )
    # Attached again: the next write needs no second full sync to get through.
    await writer.write_card(username=username, content=_content(word="next", lemma="next"))

    verify_path = tmp_path / "verify" / "collection.anki2"
    verify_path.parent.mkdir(parents=True)
    verify = Collection(str(verify_path))
    try:
        _sync_to_steady_state(
            verify, username=username, password=password, endpoint=server.endpoint
        )
        assert verify.note_count() == 2
        note = verify.get_card(card_id).note()
        assert note["ExerciseFront"] == "<p>Frage</p>"
        names = {verify.get_card(c).template()["name"] for c in verify.card_ids_of_note(note.id)}
        assert "Übung" in names
    finally:
        verify.close()
