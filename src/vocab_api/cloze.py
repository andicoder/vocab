import re


def mask_word_in_sentence(*, word: str, sentence: str) -> str | None:
    """Replace the first case-insensitive whole-word occurrence of `word` in
    `sentence` with `___`. Returns None if the word does not appear as a
    whole word in the sentence."""
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    if pattern.search(sentence) is None:
        return None
    return pattern.sub("___", sentence, count=1)
