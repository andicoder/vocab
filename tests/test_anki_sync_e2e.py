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
from anki.sync_pb2 import SyncCollectionResponse

from vocab_api.anki_sync import AnkiSyncWriter
from vocab_api.anki_writer import VocabCardContent


class _SyncServer(NamedTuple):
    endpoint: str
    username: str
    password: str


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
    proc = subprocess.Popen(
        [sys.executable, "-m", "anki.syncserver"],
        env={
            **os.environ,
            "SYNC_USER1": f"{username}:{password}",
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
