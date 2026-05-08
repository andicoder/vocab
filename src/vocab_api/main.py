from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from . import __version__
from .audio import EdgeTtsClient, make_storage_from_settings
from .config import settings
from .db import SessionLocal
from .gemini import GeminiClient
from .routes import audio, translate, vocab
from .worker import run_worker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    http_client = httpx.AsyncClient(timeout=settings.gemini_timeout_s)
    gemini = GeminiClient(
        http=http_client,
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
    )
    tts = EdgeTtsClient()
    storage = make_storage_from_settings()

    app.state.gemini = gemini
    app.state.tts = tts
    app.state.storage = storage

    try:
        if settings.gemini_api_key:
            async with run_worker(
                session_factory=SessionLocal,
                gemini=gemini,
                tts=tts,
                storage=storage,
                voice=settings.audio_voice,
            ):
                yield
        else:
            yield
    finally:
        await http_client.aclose()


app = FastAPI(title="vocab-api", version=__version__, lifespan=lifespan)
app.include_router(vocab.router)
app.include_router(translate.router)
app.include_router(audio.router)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
