import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from alembic import command
from vocab_api.config import settings

_ROOT = Path(__file__).resolve().parents[1]
_TABLES = ("entry", "user", "translation_cache", "audio_cache")
_DB_FIXTURES = frozenset({"db_session", "http_client"})

_migrations_done = False


@pytest.fixture(autouse=True)
def _isolate_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests must be deterministic regardless of a developer's local .env.
    # When `gemini_api_key` is non-empty, routes like POST /vocab call the
    # real API synchronously and persist a cache row — which then collides
    # with seeded fixtures (`uq_translation_cache_word_sentence_lang`).
    # Default the key to empty for every test; tests that need to exercise
    # the active-Gemini code path opt in via their own monkeypatch (see
    # test_ui.py for the pattern).
    monkeypatch.setattr(settings, "gemini_api_key", "")


def _needs_db(request: pytest.FixtureRequest) -> bool:
    return bool(_DB_FIXTURES.intersection(request.fixturenames))


async def _truncate_all() -> None:
    targets = ", ".join(f'{settings.db_schema}."{t}"' for t in _TABLES)
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {targets} RESTART IDENTITY CASCADE"))
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _db_setup(request: pytest.FixtureRequest) -> None:
    # Only touch Postgres for tests that actually use it. Pure unit tests
    # (e.g. test_cloze.py) need neither the migration run nor the truncate,
    # and forcing a connection would make them fail on a dev box without
    # the dev DB running.
    if not _needs_db(request):
        return
    global _migrations_done
    if not _migrations_done:
        cfg = Config(_ROOT / "alembic.ini")
        cfg.set_main_option("script_location", str(_ROOT / "alembic"))
        cfg.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(cfg, "head")
        _migrations_done = True
    asyncio.run(_truncate_all())


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.connect() as conn:
            outer = await conn.begin()
            session = AsyncSession(
                bind=conn,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )
            try:
                yield session
            finally:
                await session.close()
                await outer.rollback()
    finally:
        await engine.dispose()
