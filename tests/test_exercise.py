"""Tests for exercise.py pure helpers — no DB required."""

from types import SimpleNamespace

from vocab_api.exercise import active_exercise, exercise_pool


def _entry(**overrides):
    defaults = {
        "id": 1,
        "lemma": "expedition",
        "translation": "die Expedition",
        "sense_label": None,
        "ipa": "/ˌɛkspɪˈdɪʃən/",
        "sentence": "A grand expedition north.",
        "mc_distractors": None,
        "synonym": None,
        "antonym": None,
        "definition_en": None,
        "audio_url": None,
        "cloze_index": 0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_pool_is_empty_without_exercise_data():
    assert exercise_pool(_entry()) == []
    assert active_exercise(_entry()) is None


def test_mc_exercise_shows_all_options_and_answer_on_back():
    entry = _entry(mc_distractors="die Ausstellung · die Auslieferung · der Ausflug")
    pool = exercise_pool(entry)
    assert len(pool) == 1
    exercise = pool[0]
    for option in (
        "die Ausstellung",
        "die Auslieferung",
        "der Ausflug",
        "die Expedition",
    ):
        assert option in exercise.front
    assert "expedition" in exercise.front
    assert exercise.front.count("die Expedition") == 1
    assert "die Expedition" in exercise.back


def test_mc_options_stay_folded_until_the_learner_opens_them():
    # Recalling before seeing the options beats plain multiple choice.
    entry = _entry(mc_distractors="die Ausstellung · die Auslieferung · der Ausflug")
    front = exercise_pool(entry)[0].front
    question, _, folded = front.partition("<details>")
    assert "expedition" in question
    assert "die Expedition" not in question
    assert "<summary>" in folded
    assert "die Expedition" in folded


def test_mc_answer_position_is_deterministic_and_varies_with_rotation():
    def position(entry) -> int:
        front = exercise_pool(entry)[0].front
        options = [part.split("</li>")[0] for part in front.split("<li>")[1:]]
        return options.index("die Expedition")

    entry_a = _entry(id=10, mc_distractors="a · b · c")
    entry_b = _entry(id=10, mc_distractors="a · b · c")
    assert position(entry_a) == position(entry_b) == 10 % 4

    rotated = _entry(id=10, cloze_index=1, mc_distractors="a · b · c")
    assert position(rotated) == 11 % 4


def test_synonym_exercise_shows_synonym_and_antonym_and_asks_for_the_lemma():
    entry = _entry(synonym="journey (weniger organisiert)", antonym="homecoming")
    exercise = exercise_pool(entry)[0]
    assert "journey (weniger organisiert)" in exercise.front
    assert "homecoming" in exercise.front
    assert "expedition" not in exercise.front
    assert "expedition" in exercise.back
    assert "die Expedition" in exercise.back


def test_synonym_exercise_shows_only_the_antonym_when_no_synonym_exists():
    entry = _entry(antonym="homecoming")
    exercise = exercise_pool(entry)[0]
    assert "homecoming" in exercise.front
    assert "≈" not in exercise.front
    assert "expedition" not in exercise.front


def test_synonym_exercise_omits_antonym_when_missing():
    entry = _entry(synonym="journey")
    exercise = exercise_pool(entry)[0]
    assert "journey" in exercise.front
    assert "≠" not in exercise.front


def test_synonym_exercise_drops_hints_that_give_the_lemma_away():
    entry = _entry(lemma="desirable", synonym="appealing", antonym="undesirable (unerwünscht)")
    front = exercise_pool(entry)[0].front
    assert "appealing" in front
    assert "undesirable" not in front


def test_synonym_exercise_is_skipped_when_every_hint_gives_the_lemma_away():
    entry = _entry(lemma="toward", synonym="towards (britisch)")
    assert exercise_pool(entry) == []


def test_definition_exercise_hides_lemma_on_front():
    entry = _entry(definition_en="An organized journey undertaken for a specific purpose.")
    exercise = exercise_pool(entry)[0]
    assert "An organized journey" in exercise.front
    assert "expedition" not in exercise.front
    assert "expedition" in exercise.back
    assert "/ˌɛkspɪˈdɪʃən/" in exercise.back


def test_listening_exercise_plays_the_audio_and_hides_the_word():
    entry = _entry(audio_url="https://media.example/abc.mp3")
    exercise = exercise_pool(entry)[0]
    assert exercise.audio is True
    assert "expedition" not in exercise.front
    assert "die Expedition" not in exercise.front
    assert "expedition" in exercise.back
    assert "/ˌɛkspɪˈdɪʃən/" in exercise.back
    assert "die Expedition" in exercise.back


def test_only_the_listening_exercise_plays_audio():
    entry = _entry(
        audio_url="https://media.example/abc.mp3",
        mc_distractors="a · b · c",
        synonym="journey",
        definition_en="An organized journey.",
    )
    assert [exercise.audio for exercise in exercise_pool(entry)] == [False, False, False, True]


def test_active_exercise_cycles_through_pool_with_cloze_index():
    data = {
        "mc_distractors": "a · b · c",
        "synonym": "journey",
        "definition_en": "An organized journey.",
        "audio_url": "https://media.example/abc.mp3",
    }
    fronts = [
        active_exercise(_entry(cloze_index=i, **data)).front  # type: ignore[union-attr]
        for i in range(5)
    ]
    assert len(set(fronts[:4])) == 4, "four distinct exercise types expected"
    # MC fronts differ across rotations (answer position moves), so compare
    # the wrap-around on a position-stable type instead.
    assert fronts[2] == active_exercise(_entry(cloze_index=6, **data)).front  # type: ignore[union-attr]
