"""entry + translation_cache exercise fields (rotating Übung card)

Revision ID: b9e4d2c7a3f6
Revises: e7c2b5d8f1a4
Create Date: 2026-07-10 00:00:00.000000

Adds the exercise-card payload: multiple-choice distractors, a synonym and
antonym, a German production prompt with an English model solution, and a
monolingual English definition. Together they feed one additional rotating
"Übung" Anki template per note — the rotate-cloze job cycles the card's
content through the exercise types so every review poses a different task.

`definition_en` is the cache freshness signal for this generation — rows
from before this migration carry `definition_en = ''` and get re-translated
on next read so they pick up the exercise fields (same approach as #43/#60).

Shape mirrors #60: nullable on `entry` (NULL flags a row that predates the
exercise fields and gets backfilled on approve), NOT NULL with empty default
on `translation_cache`.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "b9e4d2c7a3f6"
down_revision: str | Sequence[str] | None = "e7c2b5d8f1a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


_COLUMNS = (
    "mc_distractors",
    "synonym",
    "antonym",
    "production_prompt",
    "production_solution",
    "definition_en",
)


def upgrade() -> None:
    for column in _COLUMNS:
        op.add_column(
            "entry",
            sa.Column(column, sa.Text(), nullable=True),
            schema=SCHEMA,
        )
    for column in _COLUMNS:
        op.add_column(
            "translation_cache",
            sa.Column(column, sa.Text(), nullable=False, server_default=""),
            schema=SCHEMA,
        )


def downgrade() -> None:
    for column in reversed(_COLUMNS):
        op.drop_column("translation_cache", column, schema=SCHEMA)
    for column in reversed(_COLUMNS):
        op.drop_column("entry", column, schema=SCHEMA)
