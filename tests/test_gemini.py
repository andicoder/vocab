import json

import httpx
import pytest

from vocab_api.gemini import GeminiClient, InventedExample, Plausibility, TranslationResult


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
            "sense_key": "noun-journey",
            "sense_label": "Reise",
            "collocations": ["go on an expedition", "Arctic expedition", "lead an expedition"],
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
        sense_key="noun-journey",
        sense_label="Reise",
        collocations=["go on an expedition", "Arctic expedition", "lead an expedition"],
    )
    # Defend against pydantic silently ignoring unknown fields — these must
    # actually be present on the model after #24 / #27.
    assert result.sense_key == "noun-journey"
    assert result.sense_label == "Reise"
    assert result.collocations == [
        "go on an expedition",
        "Arctic expedition",
        "lead an expedition",
    ]


async def test_translate_defaults_collocations_to_empty_list_when_omitted():
    # Real Gemini responses are noisy; treat a missing field as "no
    # collocations" rather than a hard failure so the worker can still
    # ship the card.
    payload = json.dumps(
        {
            "lemma": "x",
            "translation": "y",
            "alternatives": "",
            "ipa": "",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert result.collocations == []


async def test_translate_prompt_asks_for_collocations():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(
                json.dumps(
                    {
                        "lemma": "decision",
                        "translation": "die Entscheidung",
                        "alternatives": "",
                        "ipa": "",
                        "collocations": [],
                    }
                )
            ),
        )

    client, http = _make_client(handler)
    try:
        await client.translate(word="decision", sentence=None)
    finally:
        await http.aclose()

    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "collocations" in prompt
    # The prompt should anchor the shape with at least one example collocation
    # so the model doesn't return single-word entries.
    assert "make a decision" in prompt


async def test_translate_returns_extra_examples():
    payload = json.dumps(
        {
            "lemma": "take effect",
            "translation": "in Kraft treten",
            "alternatives": "",
            "ipa": "",
            "extra_examples": [
                "The new policy will take effect on Jan 1st.",
                "When does the change take effect?",
            ],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="take effect", sentence=None)
    finally:
        await http.aclose()

    assert result.extra_examples == [
        "The new policy will take effect on Jan 1st.",
        "When does the change take effect?",
    ]


async def test_translate_defaults_extra_examples_to_empty_list_when_omitted():
    # Same defensive default as collocations — a stripped-down Gemini
    # response shouldn't fail card creation.
    payload = json.dumps(
        {
            "lemma": "x",
            "translation": "y",
            "alternatives": "",
            "ipa": "",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert result.extra_examples == []


async def test_translate_prompt_asks_for_extra_examples_in_other_contexts():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(
                json.dumps(
                    {
                        "lemma": "x",
                        "translation": "y",
                        "alternatives": "",
                        "ipa": "",
                        "extra_examples": [],
                    }
                )
            ),
        )

    client, http = _make_client(handler)
    try:
        await client.translate(word="x", sentence="The source sentence.")
    finally:
        await http.aclose()

    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "extra_examples" in prompt
    # The prompt should specifically ask for sentences in different contexts
    # than the source — otherwise the model just rephrases the same scene.
    assert "different" in prompt.lower() or "other context" in prompt.lower()


async def test_translate_prompt_asks_for_sense_key_and_label():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(
                json.dumps(
                    {
                        "lemma": "train",
                        "translation": "trainieren",
                        "alternatives": "",
                        "ipa": "",
                        "sense_key": "verb-exercise",
                        "sense_label": "trainieren (sportlich)",
                    }
                )
            ),
        )

    client, http = _make_client(handler)
    try:
        await client.translate(word="train", sentence="I need to train for a marathon.")
    finally:
        await http.aclose()

    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    # Anchor the slug shape and the disambiguation label in the prompt so the
    # model returns consistent values across calls (#24).
    assert "sense_key" in prompt_text
    assert "sense_label" in prompt_text
    # Show at least one example slug so the model picks the documented shape.
    assert "verb-" in prompt_text or "noun-" in prompt_text


async def test_translate_hits_correct_endpoint_and_passes_api_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
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
    # Key must travel as a header, not a query param — query strings show up
    # in httpx exception messages and leak into pod logs (#15).
    assert "key=" not in captured["url"]
    assert "topsecret" not in captured["url"]
    assert captured["headers"].get("x-goog-api-key") == "topsecret"
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "x" in prompt_text


async def test_translate_5xx_does_not_leak_api_key_in_exception_message():
    # Regression for #15. httpx renders the full request URL inside
    # HTTPStatusError, so a logged stack trace would dump the API key if it
    # were a query parameter.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client, http = _make_client(handler, api_key="AIza-topsecret-123")
    try:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    err_text = str(excinfo.value)
    assert "AIza-topsecret-123" not in err_text
    assert "key=" not in err_text


async def test_translate_prompt_includes_part_of_speech_guidance():
    # Regression for #11: without per-PoS instructions Gemini falls back to
    # noun-shaped output ("der geniale") for adjectives/verbs.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(
                json.dumps({"lemma": "x", "translation": "y", "alternatives": "", "ipa": ""})
            ),
        )

    client, http = _make_client(handler)
    try:
        await client.translate(word="smirked", sentence="He smirked at the joke.")
    finally:
        await http.aclose()

    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "part of speech" in prompt_text.lower()
    assert "verbs" in prompt_text and "infinitive" in prompt_text
    assert "adjectives" in prompt_text
    assert "nouns" in prompt_text


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


async def test_invent_example_returns_sentence_and_cloze():
    payload = json.dumps(
        {
            "sentence": "He took the train to work.",
            "cloze_sentence": "He ___ the train to work.",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_gemini_response(payload))

    client, http = _make_client(handler)
    try:
        result = await client.invent_example(lemma="take")
    finally:
        await http.aclose()

    assert result == InventedExample(
        sentence="He took the train to work.",
        cloze_sentence="He ___ the train to work.",
    )


async def test_invent_example_prompt_includes_lemma_and_asks_for_json():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_gemini_response(json.dumps({"sentence": "x.", "cloze_sentence": "___."})),
        )

    client, http = _make_client(handler)
    try:
        await client.invent_example(lemma="expedition")
    finally:
        await http.aclose()

    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "expedition" in prompt_text
    assert "___" in prompt_text  # the prompt must instruct the model on the gap marker
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
