"""Manually-invoked smoke test against real Gemini.

Per the CLAUDE.md discipline rule "Prompt and worker-routing changes
require a real import run before merge": picks a curated set of words
from a Kindle vocab.db, runs them through `translate_with_cache` plus
a plausibility check, and prints what came back so the operator can
eyeball regressions in alt-classification, lemma resolution, or
plausibility verdicts.

Usage:
    .venv/bin/python scripts/smoke_translate.py path/to/vocab.db
    .venv/bin/python scripts/smoke_translate.py vocab.db --words toward,beneath

Cache rows for the target words are cleared first so prompt edits are
actually exercised, not the previously cached translation. Needs a
populated `VOCAB_GEMINI_API_KEY` (e.g. via `.env`) and a reachable
Postgres for the cache.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vocab_api.config import settings
from vocab_api.gemini import (
    GeminiClient,
    Plausibility,
    TranslationRequest,
    TranslationResult,
    translate_with_cache,
)
from vocab_api.kindle import KindleEntry, parse_kindle_vocab
from vocab_api.models import TranslationCache

log = logging.getLogger(__name__)

# Curated to exercise the documented bug classes. Each comment names the
# expected behavior so a regression is obvious at a glance. Extend rather
# than shrink — the cost of one extra Gemini call is negligible, the cost
# of an undetected regression is real.
DEFAULT_TARGETS: tuple[str, ...] = (
    "pebbles",  # #62 plural — expect lemma=pebble, plausibility=YES
    "dozens",  # #62 plural — expect lemma=dozen, plausibility=YES
    "smirked",  # #62 past — expect lemma=smirk, plausibility=YES
    "purses",  # #62 plural — expect lemma=purse, plausibility=YES
    "toward",  # #63 AE/BE — expect alt_priority=none (filter or prompt)
    "conscious",  # #63 compound — expect alt_priority=none
    "beneath",  # #63 legit formal — expect alt e.g. 'under' (preferred)
    "unmoored",  # #63 legit rare — expect alt (minor or preferred)
    "effortlessly",  # baseline common adverb — expect alt_priority=none
    "meticulously",  # baseline literary — expect alt_priority=none
)

_HTTP_TIMEOUT_S = 60.0
_MAX_RETRIES = 3


async def _retry[T](factory: Callable[[], Awaitable[T]], *, label: str) -> T:
    last: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await factory()
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last = e
            if attempt < _MAX_RETRIES:
                print(
                    f"  retry {attempt}/{_MAX_RETRIES} for {label}: {type(e).__name__}",
                    file=sys.stderr,
                )
                await asyncio.sleep(2 * attempt)
    assert last is not None
    raise last


def _format_result(entry: KindleEntry, tr: TranslationResult, verdict: Plausibility) -> str:
    sentence = (entry.sentence or "")[:70]
    lines = [
        f"--- {entry.word!r}",
        f"  sentence:        {sentence!r}...",
        f"  lemma:           {tr.lemma!r}",
        f"  translation:     {tr.translation!r}",
        f"  plausibility:    {verdict}",
        f"  alt_priority:    {tr.alt_priority!r}",
    ]
    if tr.alt_lemma:
        lines += [
            f"  alt_lemma:       {tr.alt_lemma!r}",
            f"  alt_reason:      {tr.alt_reason!r}",
            f"  alt_translation: {tr.alt_translation!r}",
        ]
    return "\n".join(lines)


def _print_summary(rows: list[tuple[KindleEntry, TranslationResult, Plausibility]]) -> None:
    alts = Counter(r[1].alt_priority for r in rows)
    verdicts = Counter(r[2] for r in rows)
    print(f"\nsummary: {len(rows)} words")
    print(f"  alt_priority: {dict(alts)}")
    print(f"  plausibility: {dict(verdicts)}")
    needs_review = sum(1 for _, tr, v in rows if v != "YES" or tr.alt_priority == "preferred")
    print(f"  would route to needs-review: {needs_review}/{len(rows)}")


async def _smoke(
    *,
    factory: async_sessionmaker[AsyncSession],
    gemini: GeminiClient,
    entry: KindleEntry,
) -> tuple[TranslationResult, Plausibility]:
    async def translate() -> TranslationResult:
        async with factory() as session, session.begin():
            return await translate_with_cache(
                session=session,
                cache_session_factory=factory,
                gemini=gemini,
                request=TranslationRequest(word=entry.word, sentence=entry.sentence, lang="en"),
            )

    tr = await _retry(translate, label=entry.word)
    verdict = await _retry(
        lambda: gemini.plausibility(word=tr.lemma, sentence=entry.sentence, translation=tr),
        label=f"plausibility:{entry.word}",
    )
    return tr, verdict


async def _run(kindle_db: Path, words: tuple[str, ...]) -> int:
    if not settings.gemini_api_key:
        print("VOCAB_GEMINI_API_KEY missing (set via .env or env)", file=sys.stderr)
        return 1

    entries = {e.word: e for e in parse_kindle_vocab(kindle_db)}
    selected = [entries[w] for w in words if w in entries]
    missing = [w for w in words if w not in entries]
    if missing:
        print(f"missing from {kindle_db}: {missing}\n", file=sys.stderr)
    if not selected:
        print("no overlap between word list and kindle DB", file=sys.stderr)
        return 1

    engine = create_async_engine(settings.database_url, echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s, s.begin():
            await s.execute(
                delete(TranslationCache).where(
                    TranslationCache.word.in_([e.word for e in selected])
                )
            )

        results: list[tuple[KindleEntry, TranslationResult, Plausibility]] = []
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as http:
            gemini = GeminiClient(
                http=http,
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                base_url=settings.gemini_base_url,
            )
            for entry in selected:
                try:
                    tr, verdict = await _smoke(factory=factory, gemini=gemini, entry=entry)
                except Exception as e:  # noqa: BLE001 — operator-facing tool, want all failures visible
                    print(f"--- {entry.word!r}: FAILED ({type(e).__name__}: {e})")
                    continue
                results.append((entry, tr, verdict))
                print(_format_result(entry, tr, verdict))
        _print_summary(results)
    finally:
        await engine.dispose()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kindle_db", type=Path, help="Path to a Kindle vocab.db")
    parser.add_argument(
        "--words",
        type=lambda s: tuple(w.strip() for w in s.split(",") if w.strip()),
        default=DEFAULT_TARGETS,
        help=f"Comma-separated word list (default: {len(DEFAULT_TARGETS)} curated words)",
    )
    return parser.parse_args(argv)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    return asyncio.run(_run(args.kindle_db, args.words))


if __name__ == "__main__":
    sys.exit(main())
