from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__
from .anki_writer import AnkiWriter
from .audio import EdgeTtsClient, make_storage_from_settings
from .config import settings
from .db import SessionLocal
from .gemini import GeminiClient
from .routes import audio, imports, translate, ui, vocab
from .worker import WorkerDeps, run_worker

_STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    anki_writer = AnkiWriter(
        root=Path(settings.anki_collection_root), deck_name=settings.anki_deck_name
    )

    deps = WorkerDeps(
        gemini=gemini,
        tts=tts,
        storage=storage,
        anki_writer=anki_writer,
        cache_session_factory=SessionLocal,
        voice=settings.audio_voice,
    )

    # Stash on app.state so route handlers can build their own deps from the
    # same instances (see deps.get_worker_deps).
    app.state.worker_deps = deps

    try:
        if settings.gemini_api_key:
            async with run_worker(session_factory=SessionLocal, deps=deps):
                yield
        else:
            yield
    finally:
        await http_client.aclose()


app = FastAPI(title="vocab-api", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
app.include_router(vocab.router)
app.include_router(translate.router)
app.include_router(audio.router)
app.include_router(imports.router)
app.include_router(ui.router)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
