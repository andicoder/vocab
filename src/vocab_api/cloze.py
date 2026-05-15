import logging
import re

from .gemini import GeminiClient
from .models import Entry

log = logging.getLogger(__name__)


def mask_word_in_sentence(*, word: str, sentence: str) -> str | None:
    """Replace the first case-insensitive whole-word occurrence of `word` in
    `sentence` with `___`. Returns None if the word does not appear as a
    whole word in the sentence."""
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    if pattern.search(sentence) is None:
        return None
    return pattern.sub("___", sentence, count=1)


async def populate_cloze(entry: Entry, *, gemini: GeminiClient, lemma: str) -> None:
    """Fill `entry.cloze_sentence` (and `entry.sentence` if it was empty).

    Prefers the deterministic path: mask the user-submitted surface form
    (entry.word) inside the user-submitted source sentence. Falls back to
    a Gemini-invented example only when there is no source sentence at all
    or when the surface form does not appear in the source sentence
    (a rare edge case worth a warning)."""
    if entry.sentence:
        masked = mask_word_in_sentence(word=entry.word, sentence=entry.sentence)
        if masked is not None:
            entry.cloze_sentence = masked
            return
        log.warning(
            "cloze regex miss id=%s word=%r — falling back to invented example",
            entry.id,
            entry.word,
        )

    invented = await gemini.invent_example(lemma=lemma)
    entry.sentence = invented.sentence
    entry.cloze_sentence = invented.cloze_sentence
