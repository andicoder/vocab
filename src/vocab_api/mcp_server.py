import hmac
import logging
from typing import Any

import httpx
from sqlalchemy import select

from .anki_writer import AnkiBackend
from .audio import EdgeTtsClient, make_storage_from_settings
from .config import settings
from .db import SessionLocal
from .gemini import GeminiClient
from .models import User
from .operations import ApprovalDeps, ApprovePayload
from .schemas import EntryCreate
from .vocab_service import add_entry, approve_entry, list_entries, reject_entry, translate_word
from .worker import WorkerDeps

log = logging.getLogger(__name__)

_worker_deps: WorkerDeps | None = None


def configure_mcp(deps: WorkerDeps) -> None:
    global _worker_deps
    _worker_deps = deps


def _create_deps_from_settings() -> WorkerDeps:
    """Build WorkerDeps from application settings (for stdio mode)."""
    from pathlib import Path as _Path

    from .anki_sync import AnkiSyncWriter, parse_credentials_json

    http_client = httpx.AsyncClient(timeout=settings.gemini_timeout_s)
    gemini = GeminiClient(
        http=http_client,
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
    )
    tts = EdgeTtsClient()
    storage = make_storage_from_settings()

    anki_writer: AnkiBackend
    if settings.anki_sync_url:
        anki_writer = AnkiSyncWriter(
            shadow_root=_Path(settings.anki_shadow_root),
            sync_endpoint=settings.anki_sync_url,
            credentials=parse_credentials_json(settings.anki_sync_credentials_json),
        )
    else:
        from .anki_writer import AnkiWriter

        anki_writer = AnkiWriter(root=_Path(settings.anki_collection_root))

    return WorkerDeps(
        gemini=gemini,
        tts=tts,
        storage=storage,
        anki_writer=anki_writer,
        cache_session_factory=SessionLocal,
        voice=settings.audio_voice,
    )


async def _resolve_user(username: str) -> User:
    async with SessionLocal() as session:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(username=username)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


# Deferred import: FastMCP triggers a pydantic deprecation warning during
# module import (pydantic v2 + mcp SDK compat). We import it here, after our
# own config is loaded, so the warning appears early but is harmless.
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402, I001

MCPContext = Context[Any, Any, Any]

mcp = FastMCP(
    "vocab-api",
    instructions=(
        "Self-hosted vocab + translator service. "
        "Add words you encounter in context, look up entries, manage the review queue, "
        "and get on-demand translations — all scoped to your vocab user account."
    ),
    streamable_http_path="/",
)


@mcp.tool()
async def vocab_add(
    ctx: MCPContext,
    word: str,
    sentence: str | None = None,
    source: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    """Add a word to your vocab list.

    The 80 % use case: you encounter an unknown word in context — add it with
    the surrounding sentence as example. The server translates it, generates
    example sentences and collocations, synthesises audio, and queues the entry
    for review.
    """
    _check_api_key(ctx)
    if _worker_deps is None:
        raise RuntimeError("MCP server not configured — worker deps missing")

    user = await _resolve_user(settings.mcp_username)
    payload = EntryCreate(word=word, sentence=sentence, source=source, lang=lang)

    async with SessionLocal() as session:
        entry = await add_entry(
            session=session,
            user=user,
            payload=payload,
            deps=_worker_deps,
            timeout=settings.gemini_timeout_s,
        )

    return {
        "id": entry.id,
        "word": entry.word,
        "lemma": entry.lemma,
        "translation": entry.translation,
        "status": entry.status,
        "alternatives": entry.alternatives,
        "source": entry.source,
    }


@mcp.tool()
async def vocab_list(
    ctx: MCPContext,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List vocabulary entries, newest first.

    Filter by status ('pending', 'needs-review', 'synced', 'rejected') or
    leave empty for all. Limit controls how many rows (1–200, default 20).
    """
    _check_api_key(ctx)
    limit = max(1, min(limit, 200))

    user = await _resolve_user(settings.mcp_username)
    async with SessionLocal() as session:
        entries = await list_entries(session=session, user=user, status_filter=status, limit=limit)

    return [
        {
            "id": e.id,
            "word": e.word,
            "lemma": e.lemma,
            "translation": e.translation,
            "status": e.status,
            "sentence": e.sentence,
            "source": e.source,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in entries
    ]


@mcp.tool()
async def vocab_approve(  # noqa: PLR0913 — MCP tool signature is the client interface
    ctx: MCPContext,
    entry_id: int,
    lemma: str | None = None,
    translation: str | None = None,
    alternatives: str | None = None,
    ipa: str | None = None,
) -> dict[str, Any]:
    """Approve a queued entry, creating its Anki card.

    You can override the auto-generated lemma, translation, alternatives, or
    IPA. Leave them blank to accept the worker's defaults.
    The entry must be in 'needs-review' status and must already be translated.
    """
    _check_api_key(ctx)
    if _worker_deps is None:
        raise RuntimeError("MCP server not configured — worker deps missing")

    user = await _resolve_user(settings.mcp_username)
    payload = ApprovePayload(
        lemma=lemma, translation=translation, alternatives=alternatives, ipa=ipa
    )

    async with SessionLocal() as session:
        deps = ApprovalDeps(
            storage=_worker_deps.storage,
            anki_writer=_worker_deps.anki_writer,
            gemini=_worker_deps.gemini,
            tts=_worker_deps.tts,
            cache_session_factory=_worker_deps.cache_session_factory,
            voice=_worker_deps.voice,
        )
        entry = await approve_entry(
            session=session, entry_id=entry_id, user=user, payload=payload, deps=deps
        )

    return {
        "id": entry.id,
        "word": entry.word,
        "lemma": entry.lemma,
        "translation": entry.translation,
        "status": entry.status,
        "anki_card_id": entry.anki_card_id,
    }


@mcp.tool()
async def vocab_reject(
    ctx: MCPContext,
    entry_id: int,
) -> dict[str, Any]:
    """Reject a queued entry. The word will not get an Anki card."""
    _check_api_key(ctx)

    user = await _resolve_user(settings.mcp_username)
    async with SessionLocal() as session:
        entry = await reject_entry(session=session, entry_id=entry_id, user=user)

    return {
        "id": entry.id,
        "word": entry.word,
        "status": entry.status,
    }


@mcp.tool()
async def vocab_translate(
    ctx: MCPContext,
    word: str,
    sentence: str | None = None,
) -> dict[str, Any]:
    """Translate a word on-demand without saving it.

    Useful for quick lookups when you don't want to add the word to your list.
    """
    _check_api_key(ctx)
    if _worker_deps is None:
        raise RuntimeError("MCP server not configured — worker deps missing")

    result = await translate_word(gemini=_worker_deps.gemini, word=word, sentence=sentence)

    return {
        "word": word,
        "translation": result.translation,
        "alternatives": result.alternatives,
        "ipa": result.ipa,
        "lemma": result.lemma,
        "sense_label": result.sense_label,
        "alt_lemma": result.alt_lemma,
        "alt_reason": result.alt_reason,
        "alt_translation": result.alt_translation,
    }


def _check_api_key(ctx: MCPContext) -> None:
    if not settings.mcp_api_key:
        return  # MCP server not configured — only stdio transport is active
    request = ctx.request_context.request
    headers = getattr(request, "headers", None)
    if headers is None:
        return  # stdio: API key is trusted from the process environment
    api_key: str | None = headers.get("x-api-key")
    if not api_key:
        auth = headers.get("authorization", "")
        if auth.startswith("Bearer "):
            api_key = auth.removeprefix("Bearer ")
    if not api_key or not hmac.compare_digest(api_key, settings.mcp_api_key):
        raise PermissionError("invalid or missing MCP API key")


def run() -> None:
    """Run the MCP server over stdio.

    Creates worker dependencies from application settings, configures the
    global state, and starts the FastMCP stdio transport loop.
    """
    deps = _create_deps_from_settings()
    configure_mcp(deps)
    mcp.run(transport="stdio")
