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


@pytest.fixture(scope="session", autouse=True)
def _alembic_upgrade() -> None:
    cfg = Config(_ROOT / "alembic.ini")
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")


async def _truncate_all() -> None:
    targets = ", ".join(f'{settings.db_schema}."{t}"' for t in _TABLES)
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {targets} RESTART IDENTITY CASCADE"))
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _clean_tables() -> None:
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
