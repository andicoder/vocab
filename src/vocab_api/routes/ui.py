import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..anki_writer import AnkiBackend
from ..audio import AudioStorage, TtsClient
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..deps import (
    get_anki_writer,
    get_gemini,
    get_session_factory,
    get_storage,
    get_tts,
)
from ..gemini import GeminiClient
from ..i18n import current_locale, translator_for
from ..models import Entry, User
from ..operations import (
    ApprovalDeps,
    ApprovePayload,
    apply_approve,
    apply_reject,
    import_kindle_entries,
    load_owned_entry,
)
from ..worker import WorkerDeps, process_entry

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_BOOKMARKLET_JS_PATH = Path(__file__).resolve().parent.parent / "static" / "bookmarklet.js"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

log = logging.getLogger(__name__)

router = APIRouter(tags=["ui"])

# HTTPException.detail strings raised inside operations.py, mapped to i18n keys
# so the queue UI can show a localized toast instead of a silent 4xx (#52).
_APPROVE_ERROR_KEYS: dict[str, str] = {
    "entry not yet translated": "toast.error.not_translated",
    "entry not found": "toast.error.not_found",
}


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


def _htmx_toast_response(request: Request, message: str) -> Response:
    """Render error_toast and redirect the swap to #toast so any caller's
    `hx-target`/`hx-swap` is overridden — the row stays, the toast appears."""
    response = _render(request, "partials/error_toast.html", {"message": message})
    response.headers["HX-Retarget"] = "#toast"
    response.headers["HX-Reswap"] = "innerHTML"
    return response


def _http_error_toast(request: Request, exc: HTTPException) -> Response:
    locale = current_locale(request)
    t = translator_for(locale)
    key = _APPROVE_ERROR_KEYS.get(str(exc.detail), "toast.error.generic")
    return _htmx_toast_response(request, t(key))


def _unexpected_error_toast(request: Request) -> Response:
    locale = current_locale(request)
    t = translator_for(locale)
    return _htmx_toast_response(request, t("toast.error.generic"))


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
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
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

    duplicate_of: str | None = None
    if settings.gemini_api_key:
        deps = WorkerDeps(
            gemini=gemini,
            tts=tts,
            storage=storage,
            anki_writer=anki_writer,
            cache_session_factory=session_factory,
            voice=settings.audio_voice,
        )
        try:
            async with asyncio.timeout(settings.gemini_timeout_s):
                duplicate_of = await process_entry(
                    session=session, entry=entry, user=user, deps=deps
                )
        except (TimeoutError, httpx.HTTPError):
            pass

    await session.commit()

    if duplicate_of is not None:
        return _render(
            request,
            "partials/exists_toast.html",
            {"word": word, "lemma": duplicate_of},
        )

    await session.refresh(entry)
    return _render(request, "partials/added_toast.html", {"entry": entry})


@router.post("/ui/vocab/{entry_id}/approve")
async def htmx_approve(
    request: Request,
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    storage: Annotated[AudioStorage, Depends(get_storage)],
    anki_writer: Annotated[AnkiBackend, Depends(get_anki_writer)],
    lemma: Annotated[str | None, Form()] = None,
    translation: Annotated[str | None, Form()] = None,
    alternatives: Annotated[str | None, Form()] = None,
    ipa: Annotated[str | None, Form()] = None,
) -> Response:
    try:
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
            deps=ApprovalDeps(storage=storage, anki_writer=anki_writer, voice=settings.audio_voice),
        )
        await session.commit()
    except HTTPException as exc:
        return _http_error_toast(request, exc)
    except Exception:
        log.exception("htmx_approve failed entry_id=%s user=%s", entry_id, user.username)
        return _unexpected_error_toast(request)
    return HTMLResponse("", status_code=200)


@router.post("/ui/vocab/{entry_id}/reject")
async def htmx_reject(
    request: Request,
    entry_id: int,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    try:
        entry = await load_owned_entry(session, entry_id, user)
        apply_reject(entry)
        await session.commit()
    except HTTPException as exc:
        return _http_error_toast(request, exc)
    except Exception:
        log.exception("htmx_reject failed entry_id=%s user=%s", entry_id, user.username)
        return _unexpected_error_toast(request)
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
