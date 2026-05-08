import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import aioboto3
import edge_tts
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .config import settings
from .models import AudioCache


@dataclass(frozen=True, slots=True)
class AudioRequest:
    """Identifies a specific TTS rendering: same word at same voice/lang
    yields the same MP3, and lookup in `AudioCache` keys on this triple."""

    word: str
    voice: str = "en-US-AriaNeural"
    lang: str = "en"


@dataclass(frozen=True, slots=True)
class S3Config:
    endpoint_url: str
    region: str
    bucket: str
    access_key: str
    secret_key: str
    public_url_base: str


class TtsClient(Protocol):
    async def synthesize(self, *, text: str, voice: str) -> bytes: ...


class AudioStorage(Protocol):
    async def put(self, *, key: str, data: bytes, content_type: str) -> None: ...
    async def fetch(self, key: str) -> bytes: ...
    def public_url(self, key: str) -> str: ...


class EdgeTtsClient:
    async def synthesize(self, *, text: str, voice: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for evt in communicate.stream():
            if evt["type"] == "audio":
                chunks.append(evt["data"])
        return b"".join(chunks)


class LocalDirAudioStorage:
    def __init__(self, *, root: Path, public_url_base: str) -> None:
        self._root = root
        self._public_url_base = public_url_base.rstrip("/")

    @property
    def root(self) -> Path:
        return self._root

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._write, key, data)

    def _write(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def fetch(self, key: str) -> bytes:
        return await asyncio.to_thread((self._root / key).read_bytes)

    def public_url(self, key: str) -> str:
        return f"{self._public_url_base}/{key}"


class S3AudioStorage:
    def __init__(self, config: S3Config) -> None:
        self._config = S3Config(
            endpoint_url=config.endpoint_url,
            region=config.region,
            bucket=config.bucket,
            access_key=config.access_key,
            secret_key=config.secret_key,
            public_url_base=config.public_url_base.rstrip("/"),
        )
        self._session = aioboto3.Session()

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        async with self._client() as client:
            await client.put_object(
                Bucket=self._config.bucket, Key=key, Body=data, ContentType=content_type
            )

    async def fetch(self, key: str) -> bytes:
        async with self._client() as client:
            obj = await client.get_object(Bucket=self._config.bucket, Key=key)
            return await obj["Body"].read()  # type: ignore[no-any-return]

    def public_url(self, key: str) -> str:
        return f"{self._config.public_url_base}/{key}"

    def _client(self) -> Any:
        # aioboto3 sessions are cheap; the underlying httpx connection pooling
        # is per-client, so we still create one per request. Return is `Any`
        # because aioboto3 lacks proper type stubs for its async context
        # manager.
        return self._session.client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            region_name=self._config.region,
            aws_access_key_id=self._config.access_key,
            aws_secret_access_key=self._config.secret_key,
        )


def make_storage_from_settings() -> AudioStorage:
    if settings.s3_endpoint_url:
        return S3AudioStorage(
            S3Config(
                endpoint_url=settings.s3_endpoint_url,
                region=settings.s3_region,
                bucket=settings.s3_bucket,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
                public_url_base=settings.audio_public_url_base,
            )
        )
    return LocalDirAudioStorage(
        root=Path(settings.audio_local_dir),
        public_url_base=settings.audio_public_url_base or "/audio",
    )


def audio_key(req: AudioRequest) -> str:
    """Stable filename for the cached MP3 of a given (word, voice, lang)."""
    h = hashlib.sha256(f"{req.word}|{req.voice}|{req.lang}".encode()).hexdigest()[:16]
    return f"{h}.mp3"


async def synthesize_with_cache(
    *,
    session: AsyncSession,
    cache_session_factory: async_sessionmaker[AsyncSession],
    tts: TtsClient,
    storage: AudioStorage,
    request: AudioRequest,
) -> str:
    stmt = select(AudioCache).where(
        AudioCache.word == request.word,
        AudioCache.voice == request.voice,
        AudioCache.lang == request.lang,
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached is not None:
        return storage.public_url(cached.s3_key)

    data = await tts.synthesize(text=request.word, voice=request.voice)
    key = audio_key(request)
    await storage.put(key=key, data=data, content_type="audio/mpeg")
    # Commit the cache row through a fresh session so it survives a rollback
    # of the outer entry transaction (see #6). The UNIQUE constraint makes
    # this idempotent under concurrent writes for the same request.
    try:
        async with cache_session_factory() as cache_session, cache_session.begin():
            cache_session.add(
                AudioCache(word=request.word, voice=request.voice, lang=request.lang, s3_key=key)
            )
    except IntegrityError:
        pass
    return storage.public_url(key)
