import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KindleEntry:
    word: str
    sentence: str
    source: str
    lang: str


_QUERY = """
SELECT
    w.word,
    COALESCE(latest.usage, '') AS usage,
    COALESCE(b.title, '') AS title
FROM WORDS w
LEFT JOIN (
    SELECT l1.word_key, l1.book_key, l1.usage
    FROM LOOKUPS l1
    JOIN (
        SELECT word_key, MAX(timestamp) AS ts FROM LOOKUPS GROUP BY word_key
    ) m ON m.word_key = l1.word_key AND m.ts = l1.timestamp
) latest ON latest.word_key = w.id
LEFT JOIN BOOK_INFO b ON b.id = latest.book_key
WHERE w.lang = ?
ORDER BY w.timestamp DESC
"""


def parse_kindle_vocab(db_path: Path, *, lang: str = "en") -> Iterator[KindleEntry]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for word, usage, title in conn.execute(_QUERY, (lang,)):
            yield KindleEntry(word=word, sentence=usage, source=title, lang=lang)
    finally:
        conn.close()
