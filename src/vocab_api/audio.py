import asyncio
import hashlib
from pathlib import Path
from typing import Protocol

import aioboto3
import edge_tts
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import AudioCache


class TtsClient(Protocol):
    async def synthesize(self, *, text: str, voice: str) -> bytes: ...


class AudioStorage(Protocol):
    async def put(self, *, key: str, data: bytes, content_type: str) -> None: ...
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

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._write, key, data)

    def _write(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def public_url(self, key: str) -> str:
        return f"{self._public_url_base}/{key}"


class S3AudioStorage:
    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        public_url_base: str,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._region = region
        self._bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key
        self._public_url_base = public_url_base.rstrip("/")
        self._session = aioboto3.Session()

    async def put(self, *, key: str, data: bytes, content_type: str) -> None:
        async with self._session.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
        ) as client:
            await client.put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            )

    def public_url(self, key: str) -> str:
        return f"{self._public_url_base}/{key}"


def make_storage_from_settings() -> AudioStorage:
    if settings.s3_endpoint_url:
        return S3AudioStorage(
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            public_url_base=settings.audio_public_url_base,
        )
    return LocalDirAudioStorage(
        root=Path(settings.audio_local_dir),
        public_url_base=settings.audio_public_url_base or "/audio",
    )


def _audio_key(word: str, voice: str, lang: str) -> str:
    h = hashlib.sha256(f"{word}|{voice}|{lang}".encode()).hexdigest()[:16]
    return f"{h}.mp3"


async def synthesize_with_cache(
    *,
    session: AsyncSession,
    tts: TtsClient,
    storage: AudioStorage,
    word: str,
    voice: str = "en-US-AriaNeural",
    lang: str = "en",
) -> str:
    stmt = select(AudioCache).where(
        AudioCache.word == word,
        AudioCache.voice == voice,
        AudioCache.lang == lang,
    )
    cached = (await session.execute(stmt)).scalar_one_or_none()
    if cached is not None:
        return storage.public_url(cached.s3_key)

    data = await tts.synthesize(text=word, voice=voice)
    key = _audio_key(word, voice, lang)
    await storage.put(key=key, data=data, content_type="audio/mpeg")
    session.add(AudioCache(word=word, voice=voice, lang=lang, s3_key=key))
    await session.flush()
    return storage.public_url(key)
