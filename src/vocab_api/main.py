from fastapi import FastAPI

from . import __version__
from .routes import vocab

app = FastAPI(title="vocab-api", version=__version__)

app.include_router(vocab.router)


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
