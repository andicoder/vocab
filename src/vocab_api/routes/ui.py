import asyncio
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..anki_writer import AnkiWriter
from ..audio import AudioStorage, TtsClient
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..deps import get_anki_writer, get_gemini, get_storage, get_tts
from ..gemini import GeminiClient
from ..i18n import current_locale, translator_for
from ..models import Entry, User
from ..operations import (
    ApprovePayload,
    apply_approve,
    apply_reject,
    import_kindle_entries,
    load_owned_entry,
)
from ..worker import process_entry

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_BOOKMARKLET_JS_PATH = Path(__file__).resolve().parent.parent / "static" / "bookmarklet.js"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["ui"])


def _build_bookmarklet_js(base_url: str) -> str:
    source = _BOOKMARKLET_JS_PATH.read_text(encoding="utf-8")
    source = source.replace("__BASE_URL__", base_url.rstrip("/"))
    # Only strip line comments that start at the beginning of a line (optionally
    # indented) so we don't eat the "//" inside URL string literals.
    source = re.sub(r"^\s*//[^\n]*\n", "\n", source, flags=re.MULTILINE)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"\s+", " ", source).strip()


def _render(
    request: Request, template_name: str, context: dict[str, Any] | None = None
) -> Response:
    locale = current_locale(request)
    full_context: dict[str, Any] = {
        "locale": locale,
        "t": translator_for(locale),
        **(context or {}),
    }
    return templates.TemplateResponse(request, template_name, full_context)


@router.get("/", response_class=HTMLResponse)
async def index_page(
    request: Request,
    user: Annotated[User, Depends(current_user)],
    word: Annotated[str | None, Query()] = None,
    sentence: Annotated[str | None, Query()] = None,
    source: Annotated[str | None, Query()] = None,
) -> Response:
    return _render(
        request,
        "index.html",
        {
            "active": "add",
            "prefill": {
                "word": word or "",
                "sentence": sentence or "",
                "source": source or "",
            },
        },
    )


@router.get("/bookmarklet", response_class=HTMLResponse)
async def bookmarklet_page(
    request: Request,
    user: Annotated[User, Depends(current_user)],
) -> Response:
    base_url = settings.public_base_url or str(request.base_url).rstrip("/")
    return _render(
        request,
        "bookmarklet.html",
        {
            "active": "bookmarklet",
            "base_url": base_url,
            "bookmarklet_js": _build_bookmarklet_js(base_url),
        },
    )


@router.get("/queue", response_class=HTMLResponse)
async def queue_page(
    request: Request,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    stmt = (
        select(Entry)
        .where(Entry.user_id == user.id, Entry.status == "needs-review")
        .order_by(Entry.created_at.desc())
    )
    entries = list((await session.execute(stmt)).scalars().all())
    return _render(request, "queue.html", {"active": "queue", "entries": entries})


@router.post("/ui/vocab", response_class=HTMLResponse)
async def htmx_create_entry(
    request: Request,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    gemini: Annotated[GeminiClient, Depends(get_gemini)],
    tts: Annotated[TtsClient, Depends(get_tts)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiWriter, Depends(get_anki_writer)],
    word: Annotated[str, Form()],
    sentence: Annotated[str | None, Form()] = None,
    source: Annotated[str | None, Form()] = None,
) -> Response:
    entry = Entry(
        user_id=user.id,
        word=word,
        sentence=sentence or None,
        source=source or None,
    )
    session.add(entry)
    await session.flush()

    if settings.gemini_api_key:
        try:
            async with asyncio.timeout(settings.gemini_timeout_s):
                await process_entry(
                    session=session,
                    entry=entry,
                    user=user,
                    gemini=gemini,
                    tts=tts,
                    storage=storage,
                    anki_writer=anki_writer,
                    voice=settings.audio_voice,
                )
        except (TimeoutError, httpx.HTTPError):
            pass

    await session.commit()
    await session.refresh(entry)
    return _render(request, "partials/added_toast.html", {"entry": entry})


@router.post("/ui/vocab/{entry_id}/approve")
async def htmx_approve(
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiWriter, Depends(get_anki_writer)],
    lemma: Annotated[str | None, Form()] = None,
    translation: Annotated[str | None, Form()] = None,
    alternatives: Annotated[str | None, Form()] = None,
    ipa: Annotated[str | None, Form()] = None,
) -> Response:
    entry = await load_owned_entry(session, entry_id, user)
    payload = ApprovePayload(
        lemma=lemma or None,
        translation=translation or None,
        alternatives=alternatives,
        ipa=ipa,
    )
    await apply_approve(
        entry=entry,
        payload=payload,
        user=user,
        storage=storage,
        anki_writer=anki_writer,
        voice=settings.audio_voice,
    )
    await session.commit()
    return HTMLResponse("", status_code=200)


@router.post("/ui/vocab/{entry_id}/reject")
async def htmx_reject(
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    entry = await load_owned_entry(session, entry_id, user)
    apply_reject(entry)
    await session.commit()
    return HTMLResponse("", status_code=200)


@router.post("/ui/import/kindle", response_class=HTMLResponse)
async def htmx_import_kindle(
    request: Request,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    file: Annotated[UploadFile, File()],
) -> Response:
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tf.write(data)
        tmp_path = Path(tf.name)
    try:
        added, skipped = await import_kindle_entries(session=session, user=user, db_path=tmp_path)
        await session.commit()
    finally:
        tmp_path.unlink(missing_ok=True)
    return _render(
        request,
        "partials/import_toast.html",
        {"added": added, "skipped": skipped},
    )
