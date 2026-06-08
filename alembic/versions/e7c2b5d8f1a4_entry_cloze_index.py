"""entry cloze_index for sentence rotation

Revision ID: e7c2b5d8f1a4
Revises: c1d0e8a7f5b3
Create Date: 2026-06-08 00:00:00.000000

Tracks which sentence in the pool [cloze_sentence, *extra_examples] is
currently shown on the Anki card front (#82). Starts at 0 for all rows so
existing cards keep their current sentence on the first sync; rotation
advances the index on each POST /vocab/rotate-cloze call.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "e7c2b5d8f1a4"
down_revision: str | Sequence[str] | None = "c1d0e8a7f5b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "entry",
        sa.Column("cloze_index", sa.Integer(), nullable=False, server_default="0"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("entry", "cloze_index", schema=SCHEMA)
