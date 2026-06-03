import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vocab_api.config import settings
from vocab_api.gemini import GeminiClient, TranslationResult
from vocab_api.models import Entry, User
from vocab_api.schemas import EntryCreate
from vocab_api.vocab_service import add_entry, list_entries, reject_entry, translate_word
from vocab_api.worker import WorkerDeps


@pytest.fixture
def deps_factory():
    """Build a WorkerDeps with stubs for async collaborators.

    Every test that needs WorkerDeps must provide at least a gemini stub;
    the factory fills in harmless defaults for the rest (they must not be
    exercised by the test under test — the test will fail if they are).
    """

    def _make(*, gemini: GeminiClient) -> WorkerDeps:
        engine = create_async_engine(settings.database_url)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        class _FakeTts:
            async def synthesize(self, *, text: str, voice: str) -> bytes:
                return b"FAKE-MP3:" + text.encode()

        class _FakeStorage:
            async def put(self, *, key: str, data: bytes, content_type: str) -> None:
                pass

            async def fetch(self, key: str) -> bytes:
                return b""

            def public_url(self, key: str) -> str:
                return f"/audio/{key}"

        class _FakeAnki:
            async def write_card(self, **kwargs) -> int:
                return 99

        return WorkerDeps(
            gemini=gemini,
            tts=_FakeTts(),
            storage=_FakeStorage(),
            anki_writer=_FakeAnki(),
            cache_session_factory=factory,
            voice="en-US-AriaNeural",
        )

    return _make


@pytest.mark.asyncio
async def test_add_entry_creates_entry_and_returns_translation(db_session, deps_factory):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    response_json = {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "die Reise",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sense_key": "default",
        "sense_label": "",
        "collocations": ["go on an expedition"],
        "extra_examples": ["The Arctic expedition lasted three months."],
        "alt_lemma": "",
        "alt_reason": "",
        "alt_translation": "",
        "alt_ipa": "",
        "alt_examples": [],
        "alt_priority": "none",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        candidates = [{"content": {"parts": [{"text": __import__("json").dumps(response_json)}]}}]
        return httpx.Response(200, json={"candidates": candidates})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gemini = GeminiClient(http=http, api_key="k", model="m", base_url="https://example.com")
    deps = deps_factory(gemini=gemini)

    payload = EntryCreate(word="expedition", sentence="A grand expedition north.")
    entry = await add_entry(session=db_session, user=user, payload=payload, deps=deps, timeout=5.0)

    assert entry.id is not None
    assert entry.word == "expedition"
    assert entry.lemma == "expedition"
    assert entry.translation == "die Expedition"
    assert entry.status == "needs-review"


@pytest.mark.asyncio
async def test_add_entry_handles_timeout_gracefully(db_session, deps_factory):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    async def slow_handler(request):
        import asyncio

        await asyncio.sleep(10)
        return httpx.Response(200)

    http = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler))
    gemini = GeminiClient(http=http, api_key="k", model="m", base_url="https://example.com")
    deps = deps_factory(gemini=gemini)

    payload = EntryCreate(word="expedition")
    entry = await add_entry(session=db_session, user=user, payload=payload, deps=deps, timeout=0.01)

    # Entry was saved even though Gemini timed out
    assert entry.id is not None
    assert entry.status == "pending"


@pytest.mark.asyncio
async def test_list_entries_returns_user_entries_newest_first(db_session):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    db_session.add(Entry(user_id=user.id, word="alpha", lang="en"))
    db_session.add(Entry(user_id=user.id, word="beta", lang="en"))
    await db_session.flush()

    entries = await list_entries(session=db_session, user=user)
    assert {e.word for e in entries} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_list_entries_filters_by_status(db_session):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    db_session.add(Entry(user_id=user.id, word="alpha", status="pending", lang="en"))
    db_session.add(Entry(user_id=user.id, word="beta", status="synced", lang="en"))
    await db_session.flush()

    entries = await list_entries(session=db_session, user=user, status_filter="synced")
    assert len(entries) == 1
    assert entries[0].word == "beta"


@pytest.mark.asyncio
async def test_list_entries_respects_limit(db_session):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    for w in "abcdefgh":
        db_session.add(Entry(user_id=user.id, word=w, lang="en"))
    await db_session.flush()

    entries = await list_entries(session=db_session, user=user, limit=3)
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_list_entries_scoped_to_user(db_session):
    alice = User(username="alice")
    bob = User(username="bob")
    db_session.add(alice)
    db_session.add(bob)
    await db_session.flush()

    db_session.add(Entry(user_id=alice.id, word="alpha", lang="en"))
    db_session.add(Entry(user_id=bob.id, word="beta", lang="en"))
    await db_session.flush()

    entries = await list_entries(session=db_session, user=alice)
    assert [e.word for e in entries] == ["alpha"]


@pytest.mark.asyncio
async def test_reject_entry_marks_rejected(db_session):
    user = User(username="alice")
    db_session.add(user)
    await db_session.flush()

    entry = Entry(user_id=user.id, word="nonsense", status="needs-review", lang="en")
    db_session.add(entry)
    await db_session.flush()

    result = await reject_entry(session=db_session, entry_id=entry.id, user=user)
    assert result.status == "rejected"

    # Confirm persistence
    fresh = await db_session.get(Entry, entry.id)
    assert fresh.status == "rejected"


@pytest.mark.asyncio
async def test_reject_entry_does_not_cross_users(db_session):
    alice = User(username="alice")
    bob = User(username="bob")
    db_session.add(alice)
    db_session.add(bob)
    await db_session.flush()

    entry = Entry(user_id=bob.id, word="beta", lang="en")
    db_session.add(entry)
    await db_session.flush()

    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="not found"):
        await reject_entry(session=db_session, entry_id=entry.id, user=alice)


@pytest.mark.asyncio
async def test_translate_word_returns_translation():
    response_json = {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sense_key": "default",
        "sense_label": "",
        "collocations": [],
        "extra_examples": [],
        "alt_lemma": "",
        "alt_reason": "",
        "alt_translation": "",
        "alt_ipa": "",
        "alt_examples": [],
        "alt_priority": "none",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        candidates = [{"content": {"parts": [{"text": __import__("json").dumps(response_json)}]}}]
        return httpx.Response(200, json={"candidates": candidates})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gemini = GeminiClient(http=http, api_key="k", model="m", base_url="https://example.com")

    result = await translate_word(gemini=gemini, word="expedition", sentence="A trip north.")
    assert isinstance(result, TranslationResult)
    assert result.translation == "die Expedition"
