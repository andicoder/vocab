"""entry.collocations + translation_cache.collocations

Revision ID: 42b43fa4406f
Revises: bb38bc4e9f91
Create Date: 2026-05-12 00:00:00.000000

Adds the collocations column to both entry (nullable) and
translation_cache (NOT NULL with empty default) — see #27. Stored as
the already-display-formatted " · "-joined string so the Anki template
can render it without any client-side splitting.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "42b43fa4406f"
down_revision: str | Sequence[str] | None = "bb38bc4e9f91"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "entry",
        sa.Column("collocations", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "translation_cache",
        sa.Column("collocations", sa.Text(), nullable=False, server_default=""),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("translation_cache", "collocations", schema=SCHEMA)
    op.drop_column("entry", "collocations", schema=SCHEMA)
