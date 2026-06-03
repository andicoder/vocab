from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import current_user
from ..deps import get_gemini
from ..gemini import GeminiClient, TranslationResult
from ..models import User
from ..vocab_service import translate_word

router = APIRouter(tags=["translate"])


class TranslateRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)
    sentence: str | None = None


@router.post("/translate", response_model=TranslationResult)
async def translate_on_demand(
    payload: TranslateRequest,
    user: Annotated[User, Depends(current_user)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
) -> TranslationResult:
    return await translate_word(gemini=gemini, word=payload.word, sentence=payload.sentence)
