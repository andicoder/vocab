"""user.card_direction

Revision ID: bb38bc4e9f91
Revises: 59bd47ebc232
Create Date: 2026-05-12 00:00:00.000000

Adds the per-user card_direction setting (#25). Default 'de_en' matches
today's single-card behavior so legacy users see no change without
opting in. The CHECK constraint pins the column to the three documented
values; if we ever add a fourth, both the constraint and the worker need
to learn it together.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "bb38bc4e9f91"
down_revision: str | Sequence[str] | None = "59bd47ebc232"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column(
            "card_direction",
            sa.String(),
            nullable=False,
            server_default="de_en",
        ),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_user_card_direction",
        "user",
        "card_direction IN ('de_en', 'en_de', 'both')",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint("ck_user_card_direction", "user", schema=SCHEMA, type_="check")
    op.drop_column("user", "card_direction", schema=SCHEMA)
