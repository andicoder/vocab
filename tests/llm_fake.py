"""A fake LLM gateway: answers `/v1/messages` the way the Anthropic API does."""

import json
from collections.abc import Awaitable, Callable

import anthropic
import httpx2

from vocab_api.llm import LlmClient

Handler = Callable[[httpx2.Request], httpx2.Response | Awaitable[httpx2.Response]]


def message(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def reply(text: str) -> httpx2.Response:
    return httpx2.Response(200, json=message(text))


def alias_of(request: httpx2.Request) -> str:
    return str(json.loads(request.content)["model"])


def prompt_of(request: httpx2.Request) -> str:
    return str(json.loads(request.content)["messages"][0]["content"])


def make_llm(handler: Handler) -> tuple[LlmClient, httpx2.AsyncClient]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = anthropic.AsyncAnthropic(
        base_url="https://gateway.example", api_key="k", http_client=http, max_retries=0
    )
    return LlmClient(client), http
