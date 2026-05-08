"""translation_cache

Revision ID: 3f6da27b8aec
Revises: 2eb6bf0e3517
Create Date: 2026-05-08 08:15:47.374311

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "3f6da27b8aec"
down_revision: str | Sequence[str] | None = "2eb6bf0e3517"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.create_table(
        "translation_cache",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("word", sa.Text, nullable=False),
        sa.Column("sentence_hash", sa.Text, nullable=True),
        sa.Column("lang", sa.Text, nullable=False, server_default="en"),
        sa.Column("lemma", sa.Text, nullable=False),
        sa.Column("translation", sa.Text, nullable=False),
        sa.Column("alternatives", sa.Text, nullable=False, server_default=""),
        sa.Column("ipa", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "word",
            "sentence_hash",
            "lang",
            name="uq_translation_cache_word_sentence_lang",
            postgresql_nulls_not_distinct=True,
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("translation_cache", schema=SCHEMA)
