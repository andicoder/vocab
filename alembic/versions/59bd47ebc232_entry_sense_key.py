"""entry.sense_key + per-sense unique constraint

Revision ID: 59bd47ebc232
Revises: 24119147f97b
Create Date: 2026-05-12 00:00:00.000000

Adds `sense_key` (NOT NULL, default 'default') and a nullable `sense_label`
to `entry`, then widens the per-user duplicate guard so the same lemma can
appear once per distinct meaning (see #24). `translation_cache` gets the
same two fields so cached lookups return the slug they were generated with.

The existing constraint name `uq_entry_user_lemma_lang` is dropped and
replaced with `uq_entry_user_lemma_lang_sense`. Legacy rows are backfilled
with `sense_key='default'` via the server default, which keeps them unique
under the new constraint.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "59bd47ebc232"
down_revision: str | Sequence[str] | None = "24119147f97b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "entry",
        sa.Column(
            "sense_key",
            sa.String(),
            nullable=False,
            server_default="default",
        ),
        schema=SCHEMA,
    )
    op.add_column(
        "entry",
        sa.Column("sense_label", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.drop_constraint("uq_entry_user_lemma_lang", "entry", schema=SCHEMA, type_="unique")
    op.create_unique_constraint(
        "uq_entry_user_lemma_lang_sense",
        "entry",
        ["user_id", "lemma", "lang", "sense_key"],
        schema=SCHEMA,
    )

    op.add_column(
        "translation_cache",
        sa.Column(
            "sense_key",
            sa.Text(),
            nullable=False,
            server_default="default",
        ),
        schema=SCHEMA,
    )
    op.add_column(
        "translation_cache",
        sa.Column(
            "sense_label",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_column("translation_cache", "sense_label", schema=SCHEMA)
    op.drop_column("translation_cache", "sense_key", schema=SCHEMA)
    op.drop_constraint("uq_entry_user_lemma_lang_sense", "entry", schema=SCHEMA, type_="unique")
    op.create_unique_constraint(
        "uq_entry_user_lemma_lang",
        "entry",
        ["user_id", "lemma", "lang"],
        schema=SCHEMA,
    )
    op.drop_column("entry", "sense_label", schema=SCHEMA)
    op.drop_column("entry", "sense_key", schema=SCHEMA)
