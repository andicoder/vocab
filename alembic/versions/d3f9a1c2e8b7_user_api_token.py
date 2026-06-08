"""user: add api_token column for extension bearer-token auth

Revision ID: d3f9a1c2e8b7
Revises: c1d0e8a7f5b3
Create Date: 2026-06-08 00:00:00.000000

Adds a nullable, unique api_token column to vocab.user. Generated on demand
via POST /me/token; used by the browser extension so it can authenticate
without relying on Authentik forward-auth cookies (#50).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "d3f9a1c2e8b7"
down_revision: str | Sequence[str] | None = "e7c2b5d8f1a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = settings.db_schema


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column("api_token", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.create_unique_constraint(
        "uq_user_api_token",
        "user",
        ["api_token"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint("uq_user_api_token", "user", schema=SCHEMA)
    op.drop_column("user", "api_token", schema=SCHEMA)
