import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import TranslationCache


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """Identifies a translation lookup. (word, sentence, lang) jointly key
    into the `translation_cache` table."""

    word: str
    sentence: str | None
    lang: str = "en"


Plausibility = Literal["YES", "NO", "UNCLEAR"]


class TranslationResult(BaseModel):
    lemma: str
    translation: str
    alternatives: str
    ipa: str
    # `sense_key` is the slug used to allow multiple cards per lemma — one per
    # distinct meaning, see #24. Defaults to "default" so legacy code paths
    # and tests that don't care about polysemy keep working. `sense_label` is
    # a short German hint shown on the card.
    sense_key: str = "default"
    sense_label: str = ""


class InventedExample(BaseModel):
    """A self-generated example sentence and its cloze (gap) variant.

    Used when an entry has no user-supplied source sentence — we still
    want a contextual gap sentence on the front of the card."""

    sentence: str
    cloze_sentence: str


_TRANSLATE_PROMPT = """\
Translate the following English word to German for a vocabulary flashcard.
Use the sentence to determine the part of speech and the *specific meaning*
of the English word in context, then return JSON:
- lemma: dictionary form of the English word, lowercase, no article
- translation: primary German translation in dictionary form:
    * for nouns: include article (e.g. "die Expedition")
    * for verbs: German infinitive only, no article (e.g. "springen", NOT "der Sprung")
    * for adjectives/adverbs: uninflected form, no article (e.g. "genial", NOT "der geniale")
- alternatives: comma-separated German alternatives in the same form as `translation`, may be empty
- ipa: US IPA in slashes (e.g. "/ˌɛkspɪˈdɪʃən/")
- sense_key: canonical slug for the specific meaning, lowercase ASCII,
    hyphen-separated, max 32 chars. Shape: `<part-of-speech>-<distinguisher>`,
    e.g. "verb-exercise", "noun-journey", "noun-railway", "verb-instruct".
    Use the SAME slug whenever the same meaning recurs.
- sense_label: short German hint that disambiguates this meaning from others
    (e.g. "sportlich", "Reise", "Eisenbahn"). May be empty for monosemous words.

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

_INVENT_EXAMPLE_PROMPT = """\
Invent a short, natural English example sentence (≤ 15 words) that uses the
given lemma in a clear, idiomatic context. Then return JSON:
- sentence: the example sentence, using whatever inflection of the lemma fits the sentence
- cloze_sentence: the same sentence with the inflected lemma replaced by ___

Lemma: {lemma}
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

    async def invent_example(self, *, lemma: str) -> InventedExample:
        prompt = _INVENT_EXAMPLE_PROMPT.format(lemma=lemma)
        text = await self._generate(prompt, response_mime_type="application/json")
        return InventedExample.model_validate(json.loads(text))

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
        # Key goes in a header, not the query string — httpx renders the
        # full URL inside HTTPStatusError, so a leaked stack trace would
        # otherwise dump the key into pod logs (#15).
        response = await self._http.post(url, headers={"x-goog-api-key": self._api_key}, json=body)
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
    cache_session_factory: async_sessionmaker[AsyncSession],
    gemini: GeminiClient,
    request: TranslationRequest,
) -> TranslationResult:
    sh = _sentence_hash(request.sentence)
    stmt = select(TranslationCache).where(
        TranslationCache.word == request.word,
        TranslationCache.sentence_hash.is_(sh)
        if sh is None
        else TranslationCache.sentence_hash == sh,
        TranslationCache.lang == request.lang,
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached is not None:
        return TranslationResult(
            lemma=cached.lemma,
            translation=cached.translation,
            alternatives=cached.alternatives,
            ipa=cached.ipa,
            sense_key=cached.sense_key,
            sense_label=cached.sense_label,
        )

    result = await gemini.translate(word=request.word, sentence=request.sentence)
    # Commit the cache row through a fresh session so it survives a rollback
    # of the outer entry transaction (see #6). UNIQUE constraint makes this
    # idempotent under concurrent writes for the same request.
    try:
        async with cache_session_factory() as cache_session, cache_session.begin():
            cache_session.add(
                TranslationCache(
                    word=request.word,
                    sentence_hash=sh,
                    lang=request.lang,
                    lemma=result.lemma,
                    translation=result.translation,
                    alternatives=result.alternatives,
                    ipa=result.ipa,
                    sense_key=result.sense_key,
                    sense_label=result.sense_label,
                )
            )
    except IntegrityError:
        pass
    return result
