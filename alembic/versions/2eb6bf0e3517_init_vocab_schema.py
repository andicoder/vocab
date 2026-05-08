"""init vocab schema

Revision ID: 2eb6bf0e3517
Revises:
Create Date: 2026-05-07 21:59:39.248798

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from vocab_api.config import settings

revision: str = "2eb6bf0e3517"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')

    op.create_table(
        "user",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text, nullable=False, unique=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "entry",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.BigInteger,
            sa.ForeignKey(f"{SCHEMA}.user.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("word", sa.Text, nullable=False),
        sa.Column("lemma", sa.Text, nullable=True),
        sa.Column("sentence", sa.Text, nullable=True),
        sa.Column("translation", sa.Text, nullable=True),
        sa.Column("alternatives", sa.Text, nullable=True),
        sa.Column("ipa", sa.Text, nullable=True),
        sa.Column("audio_url", sa.Text, nullable=True),
        sa.Column("source", sa.Text, nullable=True),
        sa.Column("lang", sa.Text, nullable=False, server_default="en"),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("anki_card_id", sa.BigInteger, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "meta",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.UniqueConstraint("user_id", "lemma", "lang", name="uq_entry_user_lemma_lang"),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_entry_user_status",
        "entry",
        ["user_id", "status"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("idx_entry_user_status", table_name="entry", schema=SCHEMA)
    op.drop_table("entry", schema=SCHEMA)
    op.drop_table("user", schema=SCHEMA)
