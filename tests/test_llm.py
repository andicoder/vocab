import json

import anthropic
import httpx2
import pytest

from vocab_api.llm import (
    _TRANSLATE_PROMPT,
    InventedExample,
    LlmClient,
    LlmOutputError,
    Plausibility,
    TranslationResult,
    _strip_low_value_alt,
)


def _message(text: str) -> dict:
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


def _make_client(handler, *, api_key: str = "secret"):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    client = LlmClient(
        anthropic.AsyncAnthropic(
            base_url="https://gateway.example", api_key=api_key, http_client=http, max_retries=0
        )
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

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(payload))

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
    # Real LLM responses are noisy; treat a missing field as "no
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

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert result.collocations == []


async def test_translate_prompt_asks_for_collocations():
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(
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

    prompt = captured["body"]["messages"][0]["content"]
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

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(payload))

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
    # Same defensive default as collocations — a stripped-down LLM
    # response shouldn't fail card creation.
    payload = json.dumps(
        {
            "lemma": "x",
            "translation": "y",
            "alternatives": "",
            "ipa": "",
        }
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(payload))

    client, http = _make_client(handler)
    try:
        result = await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert result.extra_examples == []


async def test_translate_prompt_asks_for_extra_examples_in_other_contexts():
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(
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

    prompt = captured["body"]["messages"][0]["content"]
    assert "extra_examples" in prompt
    # The prompt should specifically ask for sentences in different contexts
    # than the source — otherwise the model just rephrases the same scene.
    assert "different" in prompt.lower() or "other context" in prompt.lower()


async def test_translate_prompt_asks_for_sense_key_and_label():
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(
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

    prompt_text = captured["body"]["messages"][0]["content"]
    # Anchor the slug shape and the disambiguation label in the prompt so the
    # model returns consistent values across calls (#24).
    assert "sense_key" in prompt_text
    assert "sense_label" in prompt_text
    # Show at least one example slug so the model picks the documented shape.
    assert "verb-" in prompt_text or "noun-" in prompt_text


async def test_translate_asks_the_gateway_for_the_translate_alias():
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(
                json.dumps({"lemma": "x", "translation": "y", "alternatives": "", "ipa": ""})
            ),
        )

    client, http = _make_client(handler, api_key="topsecret")
    try:
        await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert captured["url"] == "https://gateway.example/v1/messages"
    assert captured["headers"]["authorization"] == "Bearer topsecret"
    # The gateway maps the task alias to a model; vocab never names one.
    assert captured["body"]["model"] == "vocab-translate"
    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"
    # Sonnet 5.5 answers any non-default temperature with a 400 (#641).
    assert "temperature" not in captured["body"]
    assert "x" in captured["body"]["messages"][0]["content"]


async def test_translate_gateway_error_does_not_leak_api_key_in_exception_message():
    # Regression for #15: a logged stack trace must not carry the key.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503, json={"error": {"message": "Service Unavailable"}})

    client, http = _make_client(handler, api_key="sk-topsecret-123")
    try:
        with pytest.raises(anthropic.APIStatusError) as excinfo:
            await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()

    assert "sk-topsecret-123" not in str(excinfo.value)


async def test_translate_prompt_includes_part_of_speech_guidance():
    # Regression for #11: without per-PoS instructions the model falls back to
    # noun-shaped output ("der geniale") for adjectives/verbs.
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(
                json.dumps({"lemma": "x", "translation": "y", "alternatives": "", "ipa": ""})
            ),
        )

    client, http = _make_client(handler)
    try:
        await client.translate(word="smirked", sentence="He smirked at the joke.")
    finally:
        await http.aclose()

    prompt_text = captured["body"]["messages"][0]["content"]
    assert "part of speech" in prompt_text.lower()
    assert "verbs" in prompt_text and "infinitive" in prompt_text
    assert "adjectives" in prompt_text
    assert "nouns" in prompt_text


async def test_translate_raises_on_http_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, text="boom")

    client, http = _make_client(handler)
    try:
        with pytest.raises(anthropic.InternalServerError):
            await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()


async def test_plausibility_yes():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message("YES"))

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
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message("Unclear — the context is ambiguous."))

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
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message("¯\\_(ツ)_/¯"))

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


async def test_plausibility_asks_for_its_alias_and_reads_past_thinking_blocks():
    captured: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        body = _message("NO")
        body["content"].insert(0, {"type": "thinking", "thinking": "", "signature": "sig"})
        return httpx2.Response(200, json=body)

    client, http = _make_client(handler)
    try:
        verdict = await client.plausibility(
            word="x",
            sentence=None,
            translation=TranslationResult(lemma="x", translation="y", alternatives="", ipa=""),
        )
    finally:
        await http.aclose()

    assert captured["body"]["model"] == "vocab-plausibility"
    assert verdict == "NO"


async def test_translate_refusal_raises_instead_of_returning_an_empty_card():
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = _message("")
        body["content"] = []
        body["stop_reason"] = "refusal"
        return httpx2.Response(200, json=body)

    client, http = _make_client(handler)
    try:
        with pytest.raises(LlmOutputError):
            await client.translate(word="x", sentence=None)
    finally:
        await http.aclose()


async def test_invent_example_returns_sentence_and_cloze():
    payload = json.dumps(
        {
            "sentence": "He took the train to work.",
            "cloze_sentence": "He ___ the train to work.",
        }
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_message(payload))

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

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json=_message(json.dumps({"sentence": "x.", "cloze_sentence": "___."})),
        )

    client, http = _make_client(handler)
    try:
        await client.invent_example(lemma="expedition")
    finally:
        await http.aclose()

    prompt_text = captured["body"]["messages"][0]["content"]
    assert "expedition" in prompt_text
    assert "___" in prompt_text  # the prompt must instruct the model on the gap marker
    assert captured["body"]["model"] == "vocab-example"
    assert captured["body"]["output_config"]["format"]["type"] == "json_schema"


# --- #63: deterministic post-filter for alt-classification false positives ---


def _alt_result(lemma: str, alt_lemma: str, *, alt_priority: str = "minor") -> TranslationResult:
    return TranslationResult(
        lemma=lemma,
        translation="(german)",
        alternatives="",
        ipa="",
        alt_lemma=alt_lemma,
        alt_reason="stylistic",
        alt_translation="(alt german)",
        alt_ipa="/alt/",
        alt_examples=["example sentence"],
        alt_priority=alt_priority,
    )


@pytest.mark.parametrize(
    ("lemma", "alt_lemma"),
    [
        # AE/BE +s suffix — dominant case from #63 (toward/towards).
        ("toward", "towards"),
        ("towards", "toward"),
        ("forward", "forwards"),
        # AE/BE suffix swaps documented in the issue ("presumably also …").
        ("color", "colour"),
        ("colour", "color"),
        ("organize", "organise"),
        ("organise", "organize"),
        ("center", "centre"),
        ("centre", "center"),
        ("analyze", "analyse"),
        ("catalog", "catalogue"),
    ],
)
def test_strip_low_value_alt_drops_ae_be_orthographic_variants(lemma: str, alt_lemma: str):
    result = _alt_result(lemma, alt_lemma)
    cleaned = _strip_low_value_alt(result)
    assert cleaned.alt_lemma == ""
    assert cleaned.alt_priority == "none"
    assert cleaned.alt_reason == ""
    assert cleaned.alt_translation == ""
    assert cleaned.alt_ipa == ""
    assert cleaned.alt_examples == []


@pytest.mark.parametrize(
    ("lemma", "alt_lemma"),
    [
        # Compound expansion via hyphen — the lemma is a component of the
        # compound, not a register variant. Documented case from #63:
        # `conscious → self-conscious`.
        ("conscious", "self-conscious"),
        ("aware", "self-aware"),
        ("aware", "self-aware-ish"),
        ("contained", "self-contained"),
        # The reverse direction too: `self-aware → aware` would also be a
        # compound relation, just dropped from the other side.
        ("self-conscious", "conscious"),
    ],
)
def test_strip_low_value_alt_drops_compound_expansions(lemma: str, alt_lemma: str):
    result = _alt_result(lemma, alt_lemma)
    cleaned = _strip_low_value_alt(result)
    assert cleaned.alt_lemma == ""
    assert cleaned.alt_priority == "none"


@pytest.mark.parametrize(
    ("lemma", "alt_lemma"),
    [
        # Legitimate alts from #63's "correct flags" list. These are
        # register/formality reductions where the meaning is preserved
        # (or close enough for a learner).
        ("beneath", "under"),
        ("deride", "ridicule"),
        ("unmoored", "disoriented"),
        ("enthralled", "captivated"),
        ("erect", "build"),
    ],
)
def test_strip_low_value_alt_preserves_legitimate_alternatives(lemma: str, alt_lemma: str):
    result = _alt_result(lemma, alt_lemma, alt_priority="minor")
    cleaned = _strip_low_value_alt(result)
    assert cleaned.alt_lemma == alt_lemma, "legitimate alts must not be dropped"
    assert cleaned.alt_priority == "minor"


def test_strip_low_value_alt_is_noop_when_no_alt():
    result = TranslationResult(
        lemma="hello",
        translation="hallo",
        alternatives="",
        ipa="",
        alt_priority="none",
    )
    assert _strip_low_value_alt(result) == result


def test_strip_low_value_alt_drops_an_alt_the_model_itself_rated_none():
    # Seen in the first smoke run through the gateway: Sonnet answered
    # `purses` with alt_priority "none" and still filled alt_lemma
    # "handbag", which the worker would have put on the card.
    result = _alt_result("purse", "handbag", alt_priority="none")

    stripped = _strip_low_value_alt(result)

    assert stripped.alt_lemma == ""
    assert stripped.alt_translation == ""
    assert stripped.alt_examples == []
    assert stripped.alt_priority == "none"


def test_translate_prompt_documents_alt_classification_negative_rules():
    # #63: pin the three negative rules into the prompt as a regression
    # guard. The deterministic filter catches AE/BE +s suffixes and
    # compound expansions, but the prompt is the only line of defense
    # against meaning-narrowing alternatives (`oblige → help`) and against
    # the less-common AE/BE patterns the filter does not enumerate.
    prompt = _TRANSLATE_PROMPT
    assert "AE/BE" in prompt or "American" in prompt, (
        "prompt must explicitly tell the model not to flag AE/BE orthographic variants"
    )
    assert "compound" in prompt.lower(), (
        "prompt must explicitly tell the model not to suggest compound expansions"
    )
    assert "strip a distinct sense" in prompt.lower() or "narrow" in prompt.lower(), (
        "prompt must warn against alternatives that strip a sense the lemma carries"
    )
    # After the post-merge calibration (#63 smoke test): the prompt must
    # also positively show what to flag, otherwise the negative rules tip
    # the model into refusing every alt (`beneath → under` got dropped).
    assert "beneath" in prompt.lower(), (
        "prompt must keep concrete positive examples (e.g. `beneath → under`) "
        "to balance the negative rules and prevent over-conservatism"
    )
