import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vocab_api.kindle import parse_kindle_vocab
from vocab_api.main import app


def _create_vocab_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE WORDS (
            id TEXT PRIMARY KEY NOT NULL,
            word TEXT NOT NULL,
            stem TEXT,
            lang TEXT NOT NULL,
            category INTEGER DEFAULT 0,
            timestamp INTEGER DEFAULT 0,
            profileid TEXT NOT NULL
        );
        CREATE TABLE LOOKUPS (
            id TEXT PRIMARY KEY NOT NULL,
            word_key TEXT NOT NULL,
            book_key TEXT NOT NULL,
            dict_key TEXT,
            pos TEXT,
            usage TEXT NOT NULL,
            timestamp INTEGER DEFAULT 0
        );
        CREATE TABLE BOOK_INFO (
            id TEXT PRIMARY KEY NOT NULL,
            asin TEXT,
            guid TEXT,
            lang TEXT NOT NULL,
            title TEXT NOT NULL,
            authors TEXT
        );
        """
    )
    return conn


def _seed_word(conn: sqlite3.Connection, *, word_id: str, word: str, lang: str = "en") -> None:
    conn.execute(
        "INSERT INTO WORDS (id, word, stem, lang, profileid) VALUES (?, ?, ?, ?, 'p')",
        (word_id, word, word, lang),
    )


def _seed_book(conn: sqlite3.Connection, *, book_id: str, title: str, lang: str = "en") -> None:
    conn.execute(
        "INSERT INTO BOOK_INFO (id, lang, title) VALUES (?, ?, ?)",
        (book_id, lang, title),
    )


def _seed_lookup(
    conn: sqlite3.Connection,
    *,
    lookup_id: str,
    word_id: str,
    book_id: str,
    usage: str,
    timestamp: int,
) -> None:
    conn.execute(
        "INSERT INTO LOOKUPS (id, word_key, book_key, dict_key, pos, usage, timestamp)"
        " VALUES (?, ?, ?, 'd', 'n', ?, ?)",
        (lookup_id, word_id, book_id, usage, timestamp),
    )


def test_parse_yields_word_with_lookup(tmp_path: Path):
    db = tmp_path / "vocab.db"
    conn = _create_vocab_db(db)
    _seed_word(conn, word_id="w1", word="expedition")
    _seed_book(conn, book_id="b1", title="Arctic Tales")
    _seed_lookup(
        conn,
        lookup_id="l1",
        word_id="w1",
        book_id="b1",
        usage="A grand expedition north.",
        timestamp=1000,
    )
    conn.commit()
    conn.close()

    entries = list(parse_kindle_vocab(db))
    assert len(entries) == 1
    e = entries[0]
    assert e.word == "expedition"
    assert e.sentence == "A grand expedition north."
    assert e.source == "Arctic Tales"
    assert e.lang == "en"


def test_parse_dedups_word_takes_latest_lookup(tmp_path: Path):
    db = tmp_path / "vocab.db"
    conn = _create_vocab_db(db)
    _seed_word(conn, word_id="w1", word="expedition")
    _seed_book(conn, book_id="b1", title="Old Book")
    _seed_book(conn, book_id="b2", title="New Book")
    _seed_lookup(
        conn,
        lookup_id="l1",
        word_id="w1",
        book_id="b1",
        usage="First sighting.",
        timestamp=1000,
    )
    _seed_lookup(
        conn,
        lookup_id="l2",
        word_id="w1",
        book_id="b2",
        usage="Latest mention.",
        timestamp=3000,
    )
    conn.commit()
    conn.close()

    entries = list(parse_kindle_vocab(db))
    assert len(entries) == 1
    assert entries[0].sentence == "Latest mention."
    assert entries[0].source == "New Book"


def test_parse_filters_by_lang(tmp_path: Path):
    db = tmp_path / "vocab.db"
    conn = _create_vocab_db(db)
    _seed_word(conn, word_id="w1", word="expedition", lang="en")
    _seed_word(conn, word_id="w2", word="hauteur", lang="fr")
    _seed_book(conn, book_id="b1", title="Book")
    _seed_lookup(conn, lookup_id="l1", word_id="w1", book_id="b1", usage="...", timestamp=1)
    _seed_lookup(conn, lookup_id="l2", word_id="w2", book_id="b1", usage="...", timestamp=2)
    conn.commit()
    conn.close()

    en = list(parse_kindle_vocab(db, lang="en"))
    fr = list(parse_kindle_vocab(db, lang="fr"))
    assert [e.word for e in en] == ["expedition"]
    assert [e.word for e in fr] == ["hauteur"]


def test_parse_handles_word_without_lookup(tmp_path: Path):
    db = tmp_path / "vocab.db"
    conn = _create_vocab_db(db)
    _seed_word(conn, word_id="w1", word="expedition")
    conn.commit()
    conn.close()

    entries = list(parse_kindle_vocab(db))
    assert len(entries) == 1
    assert entries[0].word == "expedition"
    assert entries[0].sentence == ""
    assert entries[0].source == ""


def test_parse_handles_lookup_with_unknown_book(tmp_path: Path):
    db = tmp_path / "vocab.db"
    conn = _create_vocab_db(db)
    _seed_word(conn, word_id="w1", word="expedition")
    _seed_lookup(
        conn,
        lookup_id="l1",
        word_id="w1",
        book_id="missing",
        usage="A grand expedition.",
        timestamp=1000,
    )
    conn.commit()
    conn.close()

    entries = list(parse_kindle_vocab(db))
    assert len(entries) == 1
    assert entries[0].sentence == "A grand expedition."
    assert entries[0].source == ""


@pytest.fixture
def http_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _alice() -> dict:
    return {"X-authentik-username": "alice"}


def _vocab_db_with_words(path: Path, words: list[tuple[str, str]]) -> None:
    conn = _create_vocab_db(path)
    for i, (word, sentence) in enumerate(words):
        _seed_word(conn, word_id=f"w{i}", word=word)
        _seed_book(conn, book_id=f"b{i}", title=f"Book {i}")
        _seed_lookup(
            conn,
            lookup_id=f"l{i}",
            word_id=f"w{i}",
            book_id=f"b{i}",
            usage=sentence,
            timestamp=i + 1,
        )
    conn.commit()
    conn.close()


def test_import_kindle_route_requires_auth(http_client: TestClient, tmp_path: Path):
    db = tmp_path / "vocab.db"
    _vocab_db_with_words(db, [("expedition", "...")])
    response = http_client.post("/import/kindle", files={"file": ("vocab.db", db.read_bytes())})
    assert response.status_code == 401


def test_import_kindle_route_imports_words(http_client: TestClient, tmp_path: Path):
    db = tmp_path / "vocab.db"
    _vocab_db_with_words(
        db,
        [
            ("expedition", "A grand expedition."),
            ("hauteur", "From above."),
        ],
    )

    response = http_client.post(
        "/import/kindle",
        headers=_alice(),
        files={"file": ("vocab.db", db.read_bytes())},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"added": 2, "skipped": 0}

    listing = http_client.get("/vocab", headers=_alice()).json()
    words = sorted(e["word"] for e in listing)
    assert words == ["expedition", "hauteur"]
    assert all(e["source"].startswith("Kindle:") for e in listing)


def test_import_kindle_route_skips_already_present(http_client: TestClient, tmp_path: Path):
    db = tmp_path / "vocab.db"
    _vocab_db_with_words(db, [("expedition", "A grand expedition.")])

    first = http_client.post(
        "/import/kindle",
        headers=_alice(),
        files={"file": ("vocab.db", db.read_bytes())},
    )
    second = http_client.post(
        "/import/kindle",
        headers=_alice(),
        files={"file": ("vocab.db", db.read_bytes())},
    )
    assert first.json() == {"added": 1, "skipped": 0}
    assert second.json() == {"added": 0, "skipped": 1}


def test_htmx_import_returns_toast(http_client: TestClient, tmp_path: Path):
    db = tmp_path / "vocab.db"
    _vocab_db_with_words(db, [("expedition", "...")])

    response = http_client.post(
        "/ui/import/kindle",
        headers=_alice(),
        files={"file": ("vocab.db", db.read_bytes())},
    )
    assert response.status_code == 200, response.text
    assert "1 neu" in response.text
    assert "0 übersprungen" in response.text


def test_queue_page_renders_import_form(http_client: TestClient):
    response = http_client.get("/queue", headers=_alice())
    assert response.status_code == 200
    assert 'hx-post="/ui/import/kindle"' in response.text
    assert 'name="file"' in response.text
    assert "Kindle-Import" in response.text
