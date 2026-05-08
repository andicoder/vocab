import hashlib
import json
import re
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import TranslationCache

Plausibility = Literal["YES", "NO", "UNCLEAR"]


class TranslationResult(BaseModel):
    lemma: str
    translation: str
    alternatives: str
    ipa: str


_TRANSLATE_PROMPT = """\
Translate the following English word to German for a vocabulary flashcard.
Return JSON with these fields:
- lemma: dictionary form of the English word, lowercase, no article
- translation: primary German translation including article for nouns (e.g. "die Expedition")
- alternatives: comma-separated German alternatives, may be empty
- ipa: US IPA in slashes (e.g. "/ˌɛkspɪˈdɪʃən/")

Word: {word}
{sentence_block}
"""

_PLAUSIBILITY_PROMPT = """\
Does the German translation make sense for the English word in the given context?
Reply with exactly one of: YES, NO, UNCLEAR.

English word: {word}
German translation: {translation}
{sentence_block}
"""

_VERDICT_RE = re.compile(r"\s*(YES|NO|UNCLEAR)\b")


def _sentence_block(sentence: str | None) -> str:
    return f"Sentence: {sentence}" if sentence else "(no sentence context)"


class GeminiClient:
    def __init__(self, *, http: httpx.AsyncClient, api_key: str, model: str, base_url: str) -> None:
        self._http = http
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def translate(self, *, word: str, sentence: str | None) -> TranslationResult:
        prompt = _TRANSLATE_PROMPT.format(word=word, sentence_block=_sentence_block(sentence))
        text = await self._generate(prompt, response_mime_type="application/json")
        return TranslationResult.model_validate(json.loads(text))

    async def plausibility(
        self,
        *,
        word: str,
        sentence: str | None,
        translation: TranslationResult,
    ) -> Plausibility:
        prompt = _PLAUSIBILITY_PROMPT.format(
            word=word,
            translation=translation.translation,
            sentence_block=_sentence_block(sentence),
        )
        text = await self._generate(prompt)
        match = _VERDICT_RE.match(text.upper())
        if match is None:
            return "UNCLEAR"
        verdict = match.group(1)
        assert verdict in ("YES", "NO", "UNCLEAR")
        return verdict  # type: ignore[return-value]

    async def _generate(self, prompt: str, *, response_mime_type: str | None = None) -> str:
        url = f"{self._base_url}/models/{self._model}:generateContent"
        body: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        if response_mime_type:
            body["generationConfig"] = {"responseMimeType": response_mime_type}
        response = await self._http.post(url, params={"key": self._api_key}, json=body)
        response.raise_for_status()
        data = response.json()
        return cast(str, data["candidates"][0]["content"]["parts"][0]["text"])


def _sentence_hash(sentence: str | None) -> str | None:
    if sentence is None:
        return None
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


async def translate_with_cache(
    *,
    session: AsyncSession,
    gemini: GeminiClient,
    word: str,
    sentence: str | None,
    lang: str = "en",
) -> TranslationResult:
    sh = _sentence_hash(sentence)
    stmt = select(TranslationCache).where(
        TranslationCache.word == word,
        TranslationCache.sentence_hash.is_(sh)
        if sh is None
        else TranslationCache.sentence_hash == sh,
        TranslationCache.lang == lang,
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached is not None:
        return TranslationResult(
            lemma=cached.lemma,
            translation=cached.translation,
            alternatives=cached.alternatives,
            ipa=cached.ipa,
        )

    result = await gemini.translate(word=word, sentence=sentence)
    session.add(
        TranslationCache(
            word=word,
            sentence_hash=sh,
            lang=lang,
            lemma=result.lemma,
            translation=result.translation,
            alternatives=result.alternatives,
            ipa=result.ipa,
        )
    )
    await session.flush()
    return result
