"""entry + translation_cache alt_* fields (idiomatic alternative)

Revision ID: c1d0e8a7f5b3
Revises: a4eb60988a1a
Create Date: 2026-05-18 00:00:00.000000

Adds the idiomatic-alternative payload (#60): when Gemini judges the
encountered lemma as dated/formal/regional and proposes a more common word,
the entry carries enough on its own to render the alternative on the card —
alt translation, alt IPA, alt example sentences, and the alt audio URL.

`alt_priority` (`preferred`/`minor`/`none`) drives the review routing on the
worker side: only `preferred` lands an otherwise auto-approvable entry in
needs-review. It is also the cache freshness signal — rows from before this
migration carry `alt_priority = ''` and get re-translated on next read so
they pick up the new alt_* fields (same approach as #43).

Shape mirrors #26/#27: nullable on `entry` (empty in the common case where
the lemma is fine), NOT NULL with empty default on `translation_cache`
(rows always populated, downstream code can treat `''` as "no alt").
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from vocab_api.config import settings

revision: str = "c1d0e8a7f5b3"
down_revision: str | Sequence[str] | None = "a4eb60988a1a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = settings.db_schema


_ENTRY_COLUMNS = (
    "alt_lemma",
    "alt_reason",
    "alt_translation",
    "alt_ipa",
    "alt_examples",
    "alt_audio_url",
    "alt_priority",
)

# `alt_audio_url` lives only on entry — audio dedup is handled by
# `audio_cache` keyed on (word, voice, lang). The cache columns mirror the
# entry's text fields plus `alt_priority` (the freshness signal).
_CACHE_COLUMNS = (
    "alt_lemma",
    "alt_reason",
    "alt_translation",
    "alt_ipa",
    "alt_examples",
    "alt_priority",
)


def upgrade() -> None:
    for column in _ENTRY_COLUMNS:
        op.add_column(
            "entry",
            sa.Column(column, sa.Text(), nullable=True),
            schema=SCHEMA,
        )
    for column in _CACHE_COLUMNS:
        op.add_column(
            "translation_cache",
            sa.Column(column, sa.Text(), nullable=False, server_default=""),
            schema=SCHEMA,
        )


def downgrade() -> None:
    for column in reversed(_CACHE_COLUMNS):
        op.drop_column("translation_cache", column, schema=SCHEMA)
    for column in reversed(_ENTRY_COLUMNS):
        op.drop_column("entry", column, schema=SCHEMA)
