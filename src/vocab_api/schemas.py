from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EntryCreate(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)
    sentence: str | None = None
    source: str | None = None
    lang: str = Field(default="en", min_length=2, max_length=8)


class EntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    word: str
    lemma: str | None
    sentence: str | None
    translation: str | None
    alternatives: str | None
    ipa: str | None
    audio_url: str | None
    source: str | None
    lang: str
    status: str
    anki_card_id: int | None
    created_at: datetime
    approved_at: datetime | None
    synced_at: datetime | None


class SettingsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    card_direction: Literal["de_en", "en_de", "both"]


class SettingsUpdate(BaseModel):
    card_direction: Literal["de_en", "en_de", "both"] | None = None


class TokenRead(BaseModel):
    token: str
