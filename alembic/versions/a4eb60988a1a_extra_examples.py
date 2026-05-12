"""entry.extra_examples + translation_cache.extra_examples

Revision ID: a4eb60988a1a
Revises: 42b43fa4406f
Create Date: 2026-05-12 00:00:00.000000

Adds extra example sentences (#26) — 1–2 additional sentences using the
lemma in a different context than the source sentence, shown on the
card back beneath the source. Same shape as #27: nullable on entry,
NOT NULL with empty default on translation_cache. Stored `<br>`-joined
so Anki renders each example on its own line without client splitting.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "a4eb60988a1a"
down_revision: str | Sequence[str] | None = "42b43fa4406f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "entry",
        sa.Column("extra_examples", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "translation_cache",
        sa.Column("extra_examples", sa.Text(), nullable=False, server_default=""),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("translation_cache", "extra_examples", schema=SCHEMA)
    op.drop_column("entry", "extra_examples", schema=SCHEMA)
