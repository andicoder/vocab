import json

import httpx
import pytest

from vocab_api.gemini import GeminiClient, Plausibility, TranslationResult


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def _make_client(handler, *, api_key: str = "secret", model: str = "gemini-2.5-flash-lite"):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    client = GeminiClient(
        http=http,
        api_key=api_key,
        model=model,
        base_url="https://example.com/v1beta",
    )
    return client, http


async def test_translate_returns_parsed_result():
    payload = json.dumps(
        {
            "lemma": "expedition",
            "translation": "die Expedition",
            "alternatives": "der Forschungsausflug, die Reise",
            "ipa": "/ˌɛkspɪˈdɪʃən/",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="expedition", sentence="A grand expedition north.")
    finally:
        await http.aclose()

    assert result == TranslationResult(
        lemma="expedition",
        translation="die Expedition",
        alternatives="der Forschungsausflug, die Reise",
        ipa="/ˌɛkspɪˈdɪʃən/",
    )


async def test_translate_hits_correct_endpoint_and_passes_api_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(
                json.dumps({"lemma": "x", "translation": "y", "alternatives": "", "ipa": ""})
            ),
        )

    client, http = _make_client(handler, api_key="topsecret", model="gemini-2.5-flash-lite")
    try:
        await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert captured["url"].startswith(
        "https://example.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    )
    assert "key=topsecret" in captured["url"]
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "x" in prompt_text


async def test_translate_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client, http = _make_client(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()


async def test_plausibility_yes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response("YES"))

    client, http = _make_client(handler)
    try:
        verdict: Plausibility = await client.plausibility(
            word="expedition",
            sentence="A grand expedition north.",
            translation=TranslationResult(
                lemma="expedition",
                translation="die Expedition",
                alternatives="",
                ipa="",
            ),
        )
    finally:
        await http.aclose()

    assert verdict == "YES"


async def test_plausibility_extracts_verdict_from_prose():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response("Unclear — the context is ambiguous."))

    client, http = _make_client(handler)
    try:
        verdict = await client.plausibility(
            word="bank",
            sentence=None,
            translation=TranslationResult(
                lemma="bank", translation="die Bank", alternatives="", ipa=""
            ),
        )
    finally:
        await http.aclose()

    assert verdict == "UNCLEAR"


async def test_plausibility_unparseable_falls_back_to_unclear():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response("¯\\_(ツ)_/¯"))

    client, http = _make_client(handler)
    try:
        verdict = await client.plausibility(
            word="x",
            sentence=None,
            translation=TranslationResult(lemma="x", translation="y", alternatives="", ipa=""),
        )
    finally:
        await http.aclose()

    assert verdict == "UNCLEAR"
