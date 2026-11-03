"""drop the production exercise columns

Revision ID: c4a7e1f9b2d8
Revises: b9e4d2c7a3f6
Create Date: 2026-11-03 00:00:00.000000

The Übung card no longer asks the learner to write a free English sentence:
it was too slow for an Anki review and had no single answer to grade
against. The listening exercise that replaces it needs no stored payload.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "c4a7e1f9b2d8"
down_revision: str | Sequence[str] | None = "b9e4d2c7a3f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


_COLUMNS = ("production_prompt", "production_solution")


def upgrade() -> None:
    for column in _COLUMNS:
        op.drop_column("translation_cache", column, schema=SCHEMA)
        op.drop_column("entry", column, schema=SCHEMA)


def downgrade() -> None:
    for column in _COLUMNS:
        op.add_column(
            "entry",
            sa.Column(column, sa.Text(), nullable=True),
            schema=SCHEMA,
        )
        op.add_column(
            "translation_cache",
            sa.Column(column, sa.Text(), nullable=False, server_default=""),
            schema=SCHEMA,
        )
