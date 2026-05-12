from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .config import settings


class Base(DeclarativeBase):
    metadata = MetaData(schema=settings.db_schema)


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # `card_direction` controls which Anki templates the worker materialises
    # per note (see #25). 'de_en' = production (German hint → recall English,
    # default per Wozniak's 20 Rules), 'en_de' = recognition only, 'both' = two
    # cards per note. Stored as a free-form string with a CHECK constraint at
    # the DB level; the column is small enough that an enum migration would
    # add more pain than value.
    card_direction: Mapped[str] = mapped_column(
        String, nullable=False, default="de_en", server_default="de_en"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    entries: Mapped[list["Entry"]] = relationship(back_populates="user", lazy="raise")


class Entry(Base):
    __tablename__ = "entry"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "lemma", "lang", "sense_key", name="uq_entry_user_lemma_lang_sense"
        ),
        Index("idx_entry_user_status", "user_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey(f"{settings.db_schema}.user.id"), nullable=False
    )
    word: Mapped[str] = mapped_column(Text, nullable=False)
    lemma: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    cloze_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    translation: Mapped[str | None] = mapped_column(Text, nullable=True)
    alternatives: Mapped[str | None] = mapped_column(Text, nullable=True)
    ipa: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    # `sense_key` lets the same lemma exist multiple times per user — one row
    # per distinct meaning (see #24). "default" is the slug used for legacy
    # rows that pre-date polysemy support.
    sense_key: Mapped[str] = mapped_column(
        String, nullable=False, default="default", server_default="default"
    )
    sense_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    lang: Mapped[str] = mapped_column(String, nullable=False, default="en", server_default="en")
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending", server_default="pending"
    )
    anki_card_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")

    user: Mapped[User] = relationship(back_populates="entries")


class TranslationCache(Base):
    __tablename__ = "translation_cache"
    __table_args__ = (
        UniqueConstraint(
            "word",
            "sentence_hash",
            "lang",
            name="uq_translation_cache_word_sentence_lang",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    sentence_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    lang: Mapped[str] = mapped_column(String, nullable=False, default="en", server_default="en")
    lemma: Mapped[str] = mapped_column(Text, nullable=False)
    translation: Mapped[str] = mapped_column(Text, nullable=False)
    alternatives: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    ipa: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    sense_key: Mapped[str] = mapped_column(
        Text, nullable=False, default="default", server_default="default"
    )
    sense_label: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AudioCache(Base):
    __tablename__ = "audio_cache"
    __table_args__ = (
        UniqueConstraint("word", "voice", "lang", name="uq_audio_cache_word_voice_lang"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    word: Mapped[str] = mapped_column(Text, nullable=False)
    voice: Mapped[str] = mapped_column(
        String, nullable=False, default="en-US-AriaNeural", server_default="en-US-AriaNeural"
    )
    lang: Mapped[str] = mapped_column(String, nullable=False, default="en", server_default="en")
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
