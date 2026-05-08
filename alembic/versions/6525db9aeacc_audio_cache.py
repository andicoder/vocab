"""audio_cache

Revision ID: 6525db9aeacc
Revises: 3f6da27b8aec
Create Date: 2026-05-08 08:28:04.522961

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "6525db9aeacc"
down_revision: str | Sequence[str] | None = "3f6da27b8aec"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.create_table(
        "audio_cache",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("word", sa.Text, nullable=False),
        sa.Column("voice", sa.Text, nullable=False, server_default="en-US-AriaNeural"),
        sa.Column("lang", sa.Text, nullable=False, server_default="en"),
        sa.Column("s3_key", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("word", "voice", "lang", name="uq_audio_cache_word_voice_lang"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("audio_cache", schema=SCHEMA)
