import asyncio
import json
import logging
from pathlib import Path
from typing import cast

from anki.collection import Collection
from anki.sync_pb2 import SyncAuth

from .anki_writer import CardDirection, VocabCardContent, add_vocab_note, write_media_file

log = logging.getLogger(__name__)


class AnkiSyncWriter:
    """Writes vocab cards to anki-sync-server via the Anki sync HTTP protocol.

    The pod opens a private "shadow" `collection.anki2` per user under
    `shadow_root`. anki-sync-server holds the canonical collection (which
    other clients sync against) and we never touch its file directly —
    that's what avoids the lock conflict described in #5.

    Each `write_card` call:

    1. Opens the shadow collection (no contention; only this process
       touches it).
    2. Pulls server state down via `sync_collection(sync_media=False)` so
       the shadow stays in line with notes added by other clients.
    3. Adds the vocab note locally.
    4. Pushes back via `sync_collection(sync_media=True)` so the new note
       and its audio media end up on the server.

    Concurrent writes for the same user are serialized through an
    `asyncio.Lock` because `Collection` is not thread-/coroutine-safe."""

    def __init__(
        self,
        *,
        shadow_root: Path,
        sync_endpoint: str,
        credentials: dict[str, str],
    ) -> None:
        self._shadow_root = shadow_root
        self._sync_endpoint = sync_endpoint
        self._credentials = credentials
        # Per-user hkey cache. SyncAuth is a protobuf message, lifetime
        # spans process — no expiry handling needed for the family-scale
        # deployment.
        self._auth_cache: dict[str, SyncAuth] = {}
        self._user_locks: dict[str, asyncio.Lock] = {}

    def shadow_path(self, username: str) -> Path:
        return self._shadow_root / username / "collection.anki2"

    def shadow_media_dir(self, username: str) -> Path:
        return self._shadow_root / username / "collection.media"

    def _user_lock(self, username: str) -> asyncio.Lock:
        if username not in self._user_locks:
            self._user_locks[username] = asyncio.Lock()
        return self._user_locks[username]

    async def write_card(  # noqa: PLR0913 — protocol params; bundling them into a struct would just bury them
        self,
        *,
        username: str,
        content: VocabCardContent,
        direction: CardDirection = "de_en",
        lang: str = "en",
    ) -> int:
        async with self._user_lock(username):
            return await asyncio.to_thread(self._write_and_sync, username, content, direction, lang)

    def _write_and_sync(
        self,
        username: str,
        content: VocabCardContent,
        direction: CardDirection,
        lang: str,
    ) -> int:
        if username not in self._credentials:
            raise RuntimeError(
                f"no anki-sync credentials configured for user '{username}' "
                "(set VOCAB_ANKI_SYNC_CREDENTIALS_JSON)"
            )

        col_path = self.shadow_path(username)
        col_path.parent.mkdir(parents=True, exist_ok=True)
        write_media_file(self.shadow_media_dir(username), content)

        col = Collection(str(col_path))
        try:
            auth = self._get_or_login(username, col)
            # Pull first to absorb any changes from other clients (mobile
            # review, etc.) before we mutate; otherwise the upstream sync
            # may flag a conflict that needs a full sync to resolve.
            col.sync_collection(auth, sync_media=False)
            card_id = add_vocab_note(col, content, direction=direction, lang=lang)
            col.sync_collection(auth, sync_media=True)
            return card_id
        finally:
            col.close()

    def _get_or_login(self, username: str, col: Collection) -> SyncAuth:
        cached = self._auth_cache.get(username)
        if cached is not None:
            return cached
        password = self._credentials[username]
        auth = col.sync_login(username, password, self._sync_endpoint)
        self._auth_cache[username] = auth
        return auth


def parse_credentials_json(raw: str) -> dict[str, str]:
    """Parses VOCAB_ANKI_SYNC_CREDENTIALS_JSON into a {username: password} map."""
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("anki sync credentials must be a JSON object")
    for k, v in parsed.items():
        if not isinstance(v, str):
            raise ValueError(
                f"anki sync credentials: value for {k!r} must be a string, got {type(v).__name__}"
            )
    return cast(dict[str, str], parsed)
