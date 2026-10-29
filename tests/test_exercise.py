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
        "production_prompt": None,
        "production_solution": None,
        "definition_en": None,
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


def test_synonym_exercise_asks_for_both_when_antonym_present():
    entry = _entry(synonym="journey (weniger organisiert)", antonym="homecoming")
    exercise = exercise_pool(entry)[0]
    assert "Synonym" in exercise.front
    assert "Antonym" in exercise.front
    assert "journey" in exercise.back
    assert "homecoming" in exercise.back


def test_synonym_exercise_omits_antonym_when_missing():
    entry = _entry(synonym="journey")
    exercise = exercise_pool(entry)[0]
    assert "Synonym" in exercise.front
    assert "Antonym" not in exercise.front
    assert "journey" in exercise.back


def test_definition_exercise_hides_lemma_on_front():
    entry = _entry(definition_en="An organized journey undertaken for a specific purpose.")
    exercise = exercise_pool(entry)[0]
    assert "An organized journey" in exercise.front
    assert "expedition" not in exercise.front
    assert "expedition" in exercise.back
    assert "/ˌɛkspɪˈdɪʃən/" in exercise.back


def test_production_exercise_shows_prompt_and_solution():
    entry = _entry(
        production_prompt="Du erzählst von einer Forschungsreise.",
        production_solution="The expedition reached the Arctic.",
    )
    exercise = exercise_pool(entry)[0]
    assert "Du erzählst von einer Forschungsreise." in exercise.front
    assert "The expedition reached the Arctic." not in exercise.front
    assert "The expedition reached the Arctic." in exercise.back


def test_active_exercise_cycles_through_pool_with_cloze_index():
    data = {
        "mc_distractors": "a · b · c",
        "synonym": "journey",
        "definition_en": "An organized journey.",
        "production_prompt": "Du erzählst von einer Reise.",
        "production_solution": "The expedition went north.",
    }
    fronts = [
        active_exercise(_entry(cloze_index=i, **data)).front  # type: ignore[union-attr]
        for i in range(5)
    ]
    assert len(set(fronts[:4])) == 4, "four distinct exercise types expected"
    # MC fronts differ across rotations (answer position moves), so compare
    # the wrap-around on a position-stable type instead.
    assert fronts[2] == active_exercise(_entry(cloze_index=6, **data)).front  # type: ignore[union-attr]
