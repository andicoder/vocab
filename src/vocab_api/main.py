import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

from . import __version__
from .anki_sync import AnkiSyncWriter, parse_credentials_json
from .anki_writer import AnkiBackend, AnkiWriter
from .audio import EdgeTtsClient, make_storage_from_settings
from .config import settings
from .db import SessionLocal
from .gemini import GeminiClient
from .mcp_server import configure_mcp
from .mcp_server import mcp as _mcp
from .routes import audio, imports, translate, ui, vocab
from .routes import settings as settings_routes
from .worker import WorkerDeps, run_worker

# uvicorn installs its own handlers before our app code runs; force=True so
# our format and level take precedence for vocab_api.* loggers too.
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _make_anki_backend() -> AnkiBackend:
    """Pick the Anki backend based on settings.

    `anki_sync_url` set → `AnkiSyncWriter` over the Anki sync HTTP protocol
    (production, avoids the file-lock conflict described in #5). Otherwise
    fall back to `AnkiWriter`, the file-based path used in dev/tests."""
    if settings.anki_sync_url:
        return AnkiSyncWriter(
            shadow_root=Path(settings.anki_shadow_root),
            sync_endpoint=settings.anki_sync_url,
            credentials=parse_credentials_json(settings.anki_sync_credentials_json),
        )
    return AnkiWriter(root=Path(settings.anki_collection_root))


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
    anki_writer: AnkiBackend = _make_anki_backend()

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
    configure_mcp(deps)

    # Reset session manager so streamable_http_app() creates a fresh one each
    # lifespan cycle (the manager's run() is single-use).
    _mcp._session_manager = None
    mcp_app = _mcp.streamable_http_app()
    app.router.routes = [
        r for r in app.router.routes if not (isinstance(r, Mount) and r.path == "/mcp")
    ]
    app.mount("/mcp", mcp_app)

    try:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(_mcp.session_manager.run())
            if settings.gemini_api_key:
                await stack.enter_async_context(run_worker(session_factory=SessionLocal, deps=deps))
            yield
    finally:
        await http_client.aclose()


app = FastAPI(title="vocab-api", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
# /mcp is mounted dynamically in lifespan() — the MCP session manager is
# single-use and must be recreated each startup cycle.
app.include_router(vocab.router)
app.include_router(translate.router)
app.include_router(audio.router)
app.include_router(imports.router)
app.include_router(settings_routes.router)
app.include_router(ui.router)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
