from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vocab_api.audio import (
    LocalDirAudioStorage,
    audio_key,
    synthesize_with_cache,
)
from vocab_api.models import AudioCache


class _FakeTts:
    def __init__(self, payload: bytes = b"fake-mp3") -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, *, text: str, voice: str) -> bytes:
        self.calls.append((text, voice))
        return self.payload + b":" + text.encode()


class _FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    def public_url(self, key: str) -> str:
        return f"https://cdn.example.com/{key}"


def testaudio_key_is_stable_and_distinguishes_inputs():
    a = audio_key("expedition", "en-US-AriaNeural", "en")
    b = audio_key("expedition", "en-US-AriaNeural", "en")
    c = audio_key("expedition", "en-US-GuyNeural", "en")
    d = audio_key("expedition", "en-US-AriaNeural", "de")

    assert a == b
    assert a != c
    assert a != d
    assert a.endswith(".mp3")
    assert len(a) == len("0123456789abcdef.mp3")


async def test_local_storage_writes_file_and_creates_dirs(tmp_path: Path):
    storage = LocalDirAudioStorage(root=tmp_path / "nested" / "audio", public_url_base="/audio")
    await storage.put(key="abc.mp3", data=b"hello", content_type="audio/mpeg")

    written = tmp_path / "nested" / "audio" / "abc.mp3"
    assert written.exists()
    assert written.read_bytes() == b"hello"


def test_local_storage_public_url_strips_trailing_slash():
    s = LocalDirAudioStorage(root=Path("/tmp/x"), public_url_base="https://x.example/audio/")
    assert s.public_url("k.mp3") == "https://x.example/audio/k.mp3"


async def test_synthesize_with_cache_miss_calls_tts_and_stores(db_session: AsyncSession):
    tts = _FakeTts()
    storage = _FakeStorage()

    url = await synthesize_with_cache(
        session=db_session, tts=tts, storage=storage, word="expedition"
    )

    assert tts.calls == [("expedition", "en-US-AriaNeural")]
    assert len(storage.objects) == 1
    [(key, (data, ctype))] = storage.objects.items()
    assert ctype == "audio/mpeg"
    assert data.endswith(b":expedition")
    assert url == f"https://cdn.example.com/{key}"

    rows = (await db_session.execute(select(AudioCache))).scalars().all()
    assert len(rows) == 1
    assert rows[0].word == "expedition"
    assert rows[0].voice == "en-US-AriaNeural"
    assert rows[0].s3_key == key


async def test_synthesize_with_cache_hit_skips_tts_and_storage(db_session: AsyncSession):
    tts = _FakeTts()
    storage = _FakeStorage()

    await synthesize_with_cache(session=db_session, tts=tts, storage=storage, word="expedition")

    tts2 = _FakeTts()
    storage2 = _FakeStorage()
    url = await synthesize_with_cache(
        session=db_session, tts=tts2, storage=storage2, word="expedition"
    )

    assert tts2.calls == []
    assert storage2.objects == {}
    expected_key = audio_key("expedition", "en-US-AriaNeural", "en")
    assert url == f"https://cdn.example.com/{expected_key}"


async def test_synthesize_with_cache_distinguishes_by_voice(db_session: AsyncSession):
    tts = _FakeTts()
    storage = _FakeStorage()

    await synthesize_with_cache(
        session=db_session, tts=tts, storage=storage, word="hello", voice="en-US-AriaNeural"
    )
    await synthesize_with_cache(
        session=db_session, tts=tts, storage=storage, word="hello", voice="en-US-GuyNeural"
    )

    assert len(tts.calls) == 2
    assert len(storage.objects) == 2
    rows = (await db_session.execute(select(AudioCache))).scalars().all()
    assert len(rows) == 2
