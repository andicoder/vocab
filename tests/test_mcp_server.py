import json

import httpx
import pytest
from sqlalchemy import select

from vocab_api.config import settings
from vocab_api.gemini import GeminiClient
from vocab_api.models import Entry
from vocab_api.worker import WorkerDeps


def _empty_translation_json() -> dict:
    return {
        "lemma": "expedition",
        "translation": "die Expedition",
        "alternatives": "",
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sense_key": "default",
        "sense_label": "",
        "collocations": ["go on an expedition"],
        "extra_examples": ["The Arctic expedition lasted three months."],
        # _invent_example needs sentence + cloze_sentence
        "sentence": "They went on an expedition.",
        "cloze_sentence": "They went on an {{c1::expedition}}.",
        "alt_lemma": "",
        "alt_reason": "",
        "alt_translation": "",
        "alt_ipa": "",
        "alt_examples": [],
        "alt_priority": "none",
    }


def _gemini_responding_with(payload: dict) -> GeminiClient:
    request_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(request.content.decode(errors="replace"))
        # Adapt the response to the request so each word produces a unique lemma
        resp = dict(payload)
        try:
            body = json.loads(request.content)
            contents = body["contents"][0]["parts"][0]["text"]
            # Translate requests contain "Word:"; invent-example requests
            # contain "Lemma:". Derive a unique lemma from the request.
            for line in contents.split("\n"):
                if line.startswith("Word:"):
                    resp["lemma"] = line.removeprefix("Word:").strip()
                elif line.startswith("Lemma:"):
                    resp["lemma"] = line.removeprefix("Lemma:").strip()
                    resp["sentence"] = f"This is a test sentence for {resp['lemma']}."
                    cloze = f"This is a test sentence for {{{{c1::{resp['lemma']}}}}}."
                    resp["cloze_sentence"] = cloze
        except Exception:
            pass
        msg = {"content": {"parts": [{"text": json.dumps(resp)}]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [msg]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiClient(http=http, api_key="k", model="m", base_url="https://example.com")


def _configure_stub_deps():
    """Replace mcp_server._worker_deps with stubs for testing."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import vocab_api.mcp_server as mcp_server

    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    class FakeTts:
        async def synthesize(self, *, text: str, voice: str) -> bytes:
            return b"FAKE-MP3:" + text.encode()

    class FakeStorage:
        async def put(self, *, key: str, data: bytes, content_type: str) -> None:
            pass

        async def fetch(self, key: str) -> bytes:
            return b""

        def public_url(self, key: str) -> str:
            return f"/audio/{key}"

    class FakeAnki:
        async def write_card(self, **kwargs) -> int:
            return 42

    mcp_server._worker_deps = WorkerDeps(
        gemini=_gemini_responding_with(_empty_translation_json()),
        tts=FakeTts(),
        storage=FakeStorage(),
        anki_writer=FakeAnki(),
        cache_session_factory=factory,
        voice="en-US-AriaNeural",
    )


async def _call_tool(name: str, arguments: dict) -> dict | list:
    """Call an MCP tool by name and return its result."""

    import vocab_api.mcp_server as mcp_server

    result = await mcp_server.mcp.call_tool(name, arguments)
    # call_tool returns different shapes depending on the tool and SDK version.
    # Unwrap the actual payload regardless of the outer format.
    payload = _unwrap_call_tool_result(result)
    return payload


def _unwrap_call_tool_result(result):
    """Extract the actual tool result from whatever shape call_tool returned."""
    import json as _json

    # tuple(list[TextContent], dict): pick the structured dict
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict) and "result" in item:
                return item["result"]
            if isinstance(item, dict) and "content" in item:
                return item
        return result[1] if isinstance(result[1], dict) else result

    # list[TextContent]: parse JSON from the first text block
    if isinstance(result, list) and len(result) > 0:
        if hasattr(result[0], "text"):
            return _json.loads(result[0].text)
        # Might already be a list of dicts
        if isinstance(result[0], dict):
            return result

    # Already a dict or list — return as-is
    return result


# ── auth tests ─────────────────────────────────────────────────────────────


def test_check_api_key_trusts_stdio_transport(monkeypatch):
    monkeypatch.setattr(settings, "mcp_api_key", "secret")
    from mcp.server.fastmcp import Context as MCPContext

    # Build a minimal request-like object with no headers attribute.
    class _FakeRequest:
        pass

    class _FakeRequestContext:
        request = _FakeRequest()

    ctx = MCPContext.model_construct()
    object.__setattr__(ctx, "_request_context", _FakeRequestContext())
    from vocab_api.mcp_server import _check_api_key

    _check_api_key(ctx)  # does not raise — no headers → trusted (stdio mode)


def test_check_api_key_passes_when_no_key_configured():
    from mcp.server.fastmcp import Context as MCPContext

    ctx = MCPContext.model_construct()
    from vocab_api.mcp_server import _check_api_key

    # Empty key → early return, no error
    _check_api_key(ctx)


# ── tool tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vocab_add_creates_entry(db_session):
    _configure_stub_deps()
    result = await _call_tool("vocab_add", {"word": "expedition", "sentence": "A trip north."})
    assert result["word"] == "expedition"
    assert result["lemma"] == "expedition"
    assert result["translation"] == "die Expedition"
    assert "id" in result


@pytest.mark.asyncio
async def test_vocab_list_returns_entries(db_session):
    _configure_stub_deps()
    await _call_tool("vocab_add", {"word": "alpha"})
    await _call_tool("vocab_add", {"word": "beta"})

    result = await _call_tool("vocab_list", {"limit": 10})
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["word"] == "beta"


@pytest.mark.asyncio
async def test_vocab_list_filters_by_status(db_session):
    _configure_stub_deps()
    await _call_tool("vocab_add", {"word": "alpha"})

    result = await _call_tool("vocab_list", {"status": "needs-review"})
    assert any(e["word"] == "alpha" for e in result)
    assert all(e["status"] == "needs-review" for e in result)

    empty = await _call_tool("vocab_list", {"status": "rejected"})
    assert len(empty) == 0


@pytest.mark.asyncio
async def test_vocab_reject_marks_entry_rejected(db_session):
    _configure_stub_deps()
    added = await _call_tool("vocab_add", {"word": "nonsense"})
    result = await _call_tool("vocab_reject", {"entry_id": added["id"]})
    assert result["status"] == "rejected"


@pytest.mark.asyncio
async def test_vocab_translate_returns_translation_without_saving(db_session):
    _configure_stub_deps()
    result = await _call_tool(
        "vocab_translate", {"word": "expedition", "sentence": "A trip north."}
    )
    assert result["translation"] == "die Expedition"
    assert result["lemma"] == "expedition"

    # Confirm no entry was created
    async with db_session as session:
        count = await session.scalar(select(Entry).where(Entry.word == "expedition"))
        assert count is None


@pytest.mark.asyncio
async def test_vocab_list_respects_limit(db_session):
    _configure_stub_deps()
    for w in "abcdefgh":
        await _call_tool("vocab_add", {"word": w})

    result = await _call_tool("vocab_list", {"limit": 3})
    assert len(result) == 3


# ── HTTP transport auth tests ───────────────────────────────────────────────


def _make_http_ctx(headers: dict[str, str]):
    """Build a minimal MCPContext that looks like an HTTP request."""
    from mcp.server.fastmcp import Context as MCPContext

    class _FakeRequest:
        pass

    class _FakeRequestContext:
        request = _FakeRequest()

    fake_req = _FakeRequest()
    fake_req.headers = headers  # type: ignore[attr-defined]
    ctx = MCPContext.model_construct()
    rc = _FakeRequestContext()
    rc.request = fake_req
    object.__setattr__(ctx, "_request_context", rc)
    return ctx


def test_check_api_key_passes_correct_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "mcp_api_key", "correct-secret")
    from vocab_api.mcp_server import _check_api_key

    ctx = _make_http_ctx({"authorization": "Bearer correct-secret"})
    _check_api_key(ctx)  # must not raise


def test_check_api_key_passes_correct_x_api_key_header(monkeypatch):
    monkeypatch.setattr(settings, "mcp_api_key", "correct-secret")
    from vocab_api.mcp_server import _check_api_key

    ctx = _make_http_ctx({"x-api-key": "correct-secret"})
    _check_api_key(ctx)  # must not raise


def test_check_api_key_rejects_wrong_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "mcp_api_key", "correct-secret")
    from vocab_api.mcp_server import _check_api_key

    ctx = _make_http_ctx({"authorization": "Bearer wrong-secret"})
    with pytest.raises(PermissionError):
        _check_api_key(ctx)


def test_check_api_key_rejects_missing_auth_header(monkeypatch):
    monkeypatch.setattr(settings, "mcp_api_key", "correct-secret")
    from vocab_api.mcp_server import _check_api_key

    ctx = _make_http_ctx({})
    with pytest.raises(PermissionError):
        _check_api_key(ctx)


def test_mcp_is_mounted_at_mcp_path():
    """The FastAPI app must expose a /mcp mount for HTTP transport."""
    from starlette.routing import Mount

    from vocab_api.main import app

    mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/mcp"]
    assert mounts, "/mcp mount not found in app.routes"
