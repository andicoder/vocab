"""Tests for operations.py pure helpers — no DB required."""

from types import SimpleNamespace

from vocab_api.operations import active_cloze_sentence


def _entry(
    *,
    cloze_sentence: str,
    lemma: str = "train",
    extra_examples: str | None = None,
    cloze_index: int = 0,
):
    return SimpleNamespace(
        cloze_sentence=cloze_sentence,
        lemma=lemma,
        extra_examples=extra_examples,
        cloze_index=cloze_index,
    )


def test_returns_cloze_sentence_when_no_extras():
    assert active_cloze_sentence(_entry(cloze_sentence="The ___ was late.")) == "The ___ was late."


def test_returns_cloze_sentence_at_index_zero_with_extras():
    # extra_examples are full sentences — masking happens inside cloze_pool
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A train arrived early.<br>The train left.",
        cloze_index=0,
    )
    assert active_cloze_sentence(entry) == "The ___ was late."


def test_returns_first_extra_masked_at_index_one():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A train arrived early.<br>The train left.",
        cloze_index=1,
    )
    assert active_cloze_sentence(entry) == "A ___ arrived early."


def test_returns_second_extra_masked_at_index_two():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A train arrived early.<br>The train left.",
        cloze_index=2,
    )
    assert active_cloze_sentence(entry) == "The ___ left."


def test_wraps_around_after_last_sentence():
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A train arrived early.<br>The train left.",
        cloze_index=3,
    )
    assert active_cloze_sentence(entry) == "The ___ was late."


def test_pool_of_one_always_returns_cloze_sentence():
    entry = _entry(cloze_sentence="She ___ the window.", lemma="open", cloze_index=99)
    assert active_cloze_sentence(entry) == "She ___ the window."


def test_extra_without_lemma_dropped_from_pool():
    # If the lemma doesn't appear in an extra example it is excluded;
    # cloze_index wraps over the smaller pool rather than exposing the answer.
    entry = _entry(
        cloze_sentence="The ___ was late.",
        extra_examples="A vehicle arrived.<br>The train left.",  # first has no "train"
        cloze_index=1,
    )
    # pool = ["The ___ was late.", "The ___ left."]  (first extra dropped)
    assert active_cloze_sentence(entry) == "The ___ left."
