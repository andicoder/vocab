"""Tests for operations.py pure helpers — no DB required."""

from types import SimpleNamespace

from vocab_api.operations import active_cloze_sentence


def _entry(*, cloze_sentence: str, extra_examples: str | None = None, cloze_index: int = 0):
    return SimpleNamespace(
        cloze_sentence=cloze_sentence,
        extra_examples=extra_examples,
        cloze_index=cloze_index,
    )


def test_returns_cloze_sentence_when_no_extras():
    assert active_cloze_sentence(_entry(cloze_sentence="The ___ was late.")) == "The ___ was late."


def test_returns_cloze_sentence_at_index_zero_with_extras():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A ___ arrived.<br>The ___ left.",
        cloze_index=0,
    )
    assert active_cloze_sentence(entry) == "The ___ was late."


def test_returns_first_extra_at_index_one():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A ___ arrived.<br>The ___ left.",
        cloze_index=1,
    )
    assert active_cloze_sentence(entry) == "A ___ arrived."


def test_returns_second_extra_at_index_two():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A ___ arrived.<br>The ___ left.",
        cloze_index=2,
    )
    assert active_cloze_sentence(entry) == "The ___ left."


def test_wraps_around_after_last_sentence():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A ___ arrived.<br>The ___ left.",
        cloze_index=3,
    )
    assert active_cloze_sentence(entry) == "The ___ was late."


def test_pool_of_one_always_returns_cloze_sentence():
    entry = _entry(cloze_sentence="She ___ the window.", cloze_index=99)
    assert active_cloze_sentence(entry) == "She ___ the window."
