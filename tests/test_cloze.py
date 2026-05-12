from vocab_api.cloze import mask_word_in_sentence


def test_returns_sentence_with_word_replaced_by_blank():
    assert (
        mask_word_in_sentence(word="train", sentence="The train leaves at 8.")
        == "The ___ leaves at 8."
    )


def test_match_is_case_insensitive():
    # Source-sentence forms often start a sentence with a capital, but the
    # user submits the lemma lowercase ("dozens").
    assert (
        mask_word_in_sentence(word="dozens", sentence="Dozens of pebbles littered the path.")
        == "___ of pebbles littered the path."
    )


def test_only_first_occurrence_is_replaced():
    assert (
        mask_word_in_sentence(word="train", sentence="The train will train the dog.")
        == "The ___ will train the dog."
    )


def test_returns_none_when_word_not_in_sentence():
    assert mask_word_in_sentence(word="zebra", sentence="The train leaves at 8.") is None


def test_does_not_match_inside_other_words():
    # 'train' must not match 'trainer' or 'training' — otherwise the cloze
    # would mask the wrong word.
    assert mask_word_in_sentence(word="train", sentence="The trainer is here.") is None


def test_handles_punctuation_around_word():
    assert (
        mask_word_in_sentence(word="train", sentence="A train, the express.")
        == "A ___, the express."
    )


def test_handles_word_at_end_of_sentence():
    assert (
        mask_word_in_sentence(word="marathon", sentence="I want to train for a marathon.")
        == "I want to train for a ___."
    )
