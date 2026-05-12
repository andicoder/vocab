"""entry.cloze_sentence

Revision ID: 24119147f97b
Revises: 6525db9aeacc
Create Date: 2026-05-12 00:00:00.000000

Adds a nullable text column that stores the gap-sentence (`The ___ leaves
at 8.`) used on the front of the active-recall card (#23). Nullable
because legacy rows pre-date this concept and the worker fills it on the
next sync.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "24119147f97b"
down_revision: str | Sequence[str] | None = "6525db9aeacc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "entry",
        sa.Column("cloze_sentence", sa.Text(), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("entry", "cloze_sentence", schema=SCHEMA)
