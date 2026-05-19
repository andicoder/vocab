import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel
from sqlalchemy import ColumnElement, and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    # 2–4 typical collocations the lemma appears in (`make a decision`,
    # `tough decision`, …). Shown on the card back; the empty default keeps
    # function words and adverbs (no idiomatic collocations) noise-free.
    collocations: list[str] = []
    # 1–2 additional natural example sentences using the lemma in a context
    # *different* from the source sentence. Shown on the card back beneath
    # the source sentence (#26). Empty default for the same defensive
    # reason as collocations.
    extra_examples: list[str] = []
    # Idiomatic-alternative payload (#60). Populated only when the lemma is
    # dated/formal/rare/regional and a more common word fits. `alt_priority`
    # ∈ {"preferred","minor","none"} — "preferred" routes the entry to
    # needs-review; "minor" is shown on the card but auto-approves; "none"
    # means no alt. Defaults keep older mocks (and the empty-cache test
    # payload) compatible.
    alt_lemma: str = ""
    alt_reason: str = ""
    alt_translation: str = ""
    alt_ipa: str = ""
    alt_examples: list[str] = []
    alt_priority: str = ""


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
- collocations: 2–4 typical English collocations the lemma appears in, as a JSON
    list of short phrases (e.g. ["make a decision", "tough decision",
    "reach a decision"]). Use compact phrases, not full sentences. Return an
    empty list for function words or adverbs that have no idiomatic
    collocations.
- extra_examples: 1–2 additional natural English example sentences using the
    lemma in a *different context* than the source sentence above (if any).
    Vary the surrounding domain, register and inflection so the learner sees
    the word's breadth of use, not a paraphrase of the source. Return as a
    JSON list of strings; empty list is acceptable when no different-context
    example fits.
- alt_priority: how strongly a more common/idiomatic English alternative
    should be surfaced. One of:
    * "preferred" — the lemma is clearly dated, overly formal, rare or
        regional, and a native speaker would naturally reach for `alt_lemma`
        instead in everyday speech. Examples worth flagging: `beneath` →
        `under` (formal), `deride` → `ridicule` (formal), `unmoored` →
        `disoriented` (rare in everyday speech).
    * "minor" — `alt_lemma` is a slightly more common register or stylistic
        variant; both are perfectly acceptable. Example: `enthralled` →
        `captivated` (both literary, the latter slightly more common).
    * "none" — the lemma is in normal everyday use; no alternative is
        warranted (leave `alt_lemma` and the other alt_* fields empty).
    Default to flagging legitimately formal/rare/dated lemmas — that is
    the feature's whole point. Only prefer "none" when the lemma is
    already common everyday English OR when one of the rules below applies.
    Do NOT flag (always return "none" in these cases):
    * AE/BE orthographic variants — both spellings are standard everyday
        English; neither is dated, formal, or rare. Examples:
        `toward`/`towards`, `color`/`colour`, `organize`/`organise`,
        `center`/`centre`, `gray`/`grey`, `analyze`/`analyse`.
    * Compound expansions where the alternative would be the lemma with
        a prefix or suffix attached (e.g. `conscious` → `self-conscious`,
        `aware` → `self-aware`). These are distinct concepts, not register
        variants — flagging them misleads the learner.
    * Alternatives that strip a distinct sense the lemma carries in this
        sentence. Slight register or formality shifts where the core
        meaning is preserved are exactly what to flag. Counter-example:
        `oblige` in "he obliged" carries the specific "complied with the
        request" sense — `help` would lose that, so leave the alt empty.
        But do NOT use this rule to suppress legitimate register-only
        substitutions like `beneath → under`.
- alt_lemma: the more idiomatic English word or short phrase (lowercase,
    dictionary form). Empty string when alt_priority is "none".
- alt_reason: one short tag explaining why the original lemma was flagged
    (e.g. "dated", "formal", "regional", "rare", "literary"). Empty when
    alt_priority is "none".
- alt_translation: primary German translation of `alt_lemma`, same shape
    rules as `translation` above (article for nouns, infinitive for verbs,
    uninflected for adjectives). Empty when alt_priority is "none".
- alt_ipa: US IPA for `alt_lemma` in slashes. Empty when alt_priority is
    "none".
- alt_examples: 1–2 short English example sentences using `alt_lemma` (≤ 15
    words each, JSON list of strings). The learner needs to see the
    alternative in context; the user's original sentence still contains the
    original lemma. Empty list when alt_priority is "none".

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

# `COLLOCATION_SEPARATOR` is the on-card display separator and also the
# delimiter we use to flatten the list before persisting (translation cache,
# entry column). Picked for readability on small phone screens and for being
# very unlikely to appear inside an actual collocation.
COLLOCATION_SEPARATOR = " · "

# Extra example sentences are joined with `<br>` so Anki renders each one
# on its own line. HTML-as-separator is acceptable because Anki fields are
# already HTML; the cost of escaping is bigger than the cost of the
# `<br>` leaking through to other consumers (we have none).
EXTRA_EXAMPLE_SEPARATOR = "<br>"


def join_collocations(items: list[str]) -> str:
    return COLLOCATION_SEPARATOR.join(items)


def split_collocations(joined: str) -> list[str]:
    if not joined:
        return []
    return joined.split(COLLOCATION_SEPARATOR)


def join_extra_examples(items: list[str]) -> str:
    return EXTRA_EXAMPLE_SEPARATOR.join(items)


def split_extra_examples(joined: str) -> list[str]:
    if not joined:
        return []
    return joined.split(EXTRA_EXAMPLE_SEPARATOR)


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


def _cache_row_is_fresh() -> ColumnElement[bool]:
    """Predicate that distinguishes fresh post-#60 cache rows from stale ones.

    A row counts as fresh when it has at least one of `collocations` /
    `extra_examples` populated (#43 — migrations 42b43fa4406f /
    a4eb60988a1a backfilled pre-existing rows with `''`) AND a non-empty
    `alt_priority` (#60 — the post-idiomaticity worker always emits one
    of {'preferred','minor','none'}, so `''` reliably flags a pre-#60 row).
    Used both as the WHERE filter on the cache SELECT and — negated — as
    the WHERE on the upsert's DO UPDATE, so a stale row gets refreshed in
    place while a fresh row that lost the UNIQUE race against a concurrent
    fresh writer is left alone (preserves the previous `except
    IntegrityError pass` no-op for that race). Read/write share one
    predicate so adding a future freshness signal can't drift between the
    two paths (#64 lesson)."""
    return and_(
        ~((TranslationCache.collocations == "") & (TranslationCache.extra_examples == "")),
        TranslationCache.alt_priority != "",
    )


# AE/BE orthographic suffix pairs that pair up to the *same* word. The
# prompt is asked to filter these too (#63), but Gemini still slips them
# through occasionally — this deterministic backstop catches the common
# patterns so the learner never sees `toward → towards` as a "more
# idiomatic alternative". Single-letter variants (`gray`/`grey`,
# `defense`/`defence`) are rarer and left to the prompt.
_AE_BE_SUFFIX_PAIRS: tuple[tuple[str, str], ...] = (
    ("or", "our"),  # color/colour, behavior/behaviour
    ("ize", "ise"),  # organize/organise
    ("er", "re"),  # center/centre, theater/theatre
    ("yze", "yse"),  # analyze/analyse
    ("og", "ogue"),  # catalog/catalogue
)


def _is_ae_be_variant(lemma: str, alt: str) -> bool:
    if alt == lemma + "s" or lemma == alt + "s":
        return True
    for left, right in _AE_BE_SUFFIX_PAIRS:
        if lemma.endswith(left) and alt == lemma[: -len(left)] + right:
            return True
        if alt.endswith(left) and lemma == alt[: -len(left)] + right:
            return True
    return False


def _is_compound_expansion(lemma: str, alt: str) -> bool:
    """True when `alt` is a hyphen-joined compound containing `lemma` as
    one of its components (or vice versa). `conscious` ↔ `self-conscious`
    is the canonical case from #63 — these are different concepts, not
    register variants, and surfacing one as an alternative for the other
    misleads the learner."""
    return lemma in alt.split("-") or alt in lemma.split("-")


def _strip_low_value_alt(result: TranslationResult) -> TranslationResult:
    """Drop the idiomatic-alt payload when the (lemma, alt_lemma) pair
    matches a known false-positive pattern (#63).

    Two deterministic patterns are caught here as a backstop against the
    prompt: AE/BE orthographic variants and hyphen-compound expansions.
    Meaning-narrowing alternatives (`oblige → help`) are not detectable
    deterministically and rely on the prompt's "do not flag" rules.

    Applied both at fresh-translate write time and at cache-hit read time
    so pre-existing cache rows containing false-positive alts self-heal
    without a schema migration."""
    if not result.alt_lemma:
        return result
    if not (
        _is_ae_be_variant(result.lemma, result.alt_lemma)
        or _is_compound_expansion(result.lemma, result.alt_lemma)
    ):
        return result
    return result.model_copy(
        update={
            "alt_lemma": "",
            "alt_reason": "",
            "alt_translation": "",
            "alt_ipa": "",
            "alt_examples": [],
            "alt_priority": "none",
        }
    )


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
        _cache_row_is_fresh(),
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached is not None:
        # Filter applied at read time too so cache rows written before #63
        # self-heal: old false-positive alts get stripped on every fetch
        # without needing a schema migration or bulk re-translate.
        return _strip_low_value_alt(
            TranslationResult(
                lemma=cached.lemma,
                translation=cached.translation,
                alternatives=cached.alternatives,
                ipa=cached.ipa,
                sense_key=cached.sense_key,
                sense_label=cached.sense_label,
                collocations=split_collocations(cached.collocations),
                extra_examples=split_extra_examples(cached.extra_examples),
                alt_lemma=cached.alt_lemma,
                alt_reason=cached.alt_reason,
                alt_translation=cached.alt_translation,
                alt_ipa=cached.alt_ipa,
                alt_examples=split_extra_examples(cached.alt_examples),
                alt_priority=cached.alt_priority,
            )
        )

    result = _strip_low_value_alt(
        await gemini.translate(word=request.word, sentence=request.sentence)
    )
    values = {
        "word": request.word,
        "sentence_hash": sh,
        "lang": request.lang,
        "lemma": result.lemma,
        "translation": result.translation,
        "alternatives": result.alternatives,
        "ipa": result.ipa,
        "sense_key": result.sense_key,
        "sense_label": result.sense_label,
        "collocations": join_collocations(result.collocations),
        "extra_examples": join_extra_examples(result.extra_examples),
        "alt_lemma": result.alt_lemma,
        "alt_reason": result.alt_reason,
        "alt_translation": result.alt_translation,
        "alt_ipa": result.alt_ipa,
        "alt_examples": join_extra_examples(result.alt_examples),
        "alt_priority": result.alt_priority,
    }
    insert_stmt = pg_insert(TranslationCache).values(**values)
    refresh_cols = {
        col: insert_stmt.excluded[col]
        for col in values
        if col not in ("word", "sentence_hash", "lang")
    }
    upsert_stmt = insert_stmt.on_conflict_do_update(
        constraint="uq_translation_cache_word_sentence_lang",
        set_=refresh_cols,
        where=~_cache_row_is_fresh(),
    )
    # Commit through a fresh session so the cache row survives a rollback of
    # the outer entry transaction (#6).
    async with cache_session_factory() as cache_session, cache_session.begin():
        await cache_session.execute(upsert_stmt)
    return result
