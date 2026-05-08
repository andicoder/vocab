from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from alembic import command
from vocab_api.config import settings

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _alembic_upgrade() -> None:
    cfg = Config(_ROOT / "alembic.ini")
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")


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
