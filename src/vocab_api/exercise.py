"""Rendering for the rotating "Übung" Anki card.

Each entry carries up to four exercises (multiple choice, synonym/antonym,
monolingual definition, listening). The pool skips types whose
data is missing; `active_exercise` picks one via the entry's `cloze_index`
so the existing rotate-cloze job drives both the gap-sentence rotation and
the exercise-type rotation with a single counter.
"""

from dataclasses import dataclass

from .llm import split_collocations
from .models import Entry

_WHICH_WORD = "Welches englische Wort ist gemeint?"


@dataclass(frozen=True, slots=True)
class Exercise:
    front: str
    back: str
    # The note's Audio plays on the front; only the listening exercise asks for it.
    audio: bool = False


def exercise_pool(entry: Entry) -> list[Exercise]:
    """All renderable exercises for `entry`, in stable order."""
    pool: list[Exercise] = []
    if entry.mc_distractors and entry.translation and entry.lemma:
        pool.append(_multiple_choice(entry))
    if _word_hints(entry):
        pool.append(_synonym_antonym(entry))
    if entry.definition_en and entry.lemma:
        pool.append(_definition(entry))
    if entry.audio_url and entry.lemma:
        pool.append(_listening(entry))
    return pool


def active_exercise(entry: Entry) -> Exercise | None:
    pool = exercise_pool(entry)
    if not pool:
        return None
    return pool[entry.cloze_index % len(pool)]


def _answer_line(entry: Entry) -> str:
    label = f" <em>({entry.sense_label})</em>" if entry.sense_label else ""
    return f"<p><strong>{entry.translation}</strong>{label}</p>"


def _multiple_choice(entry: Entry) -> Exercise:
    options = split_collocations(entry.mc_distractors or "")
    # Deterministic but rotation-dependent answer position: no random (a
    # re-render must be reproducible) and no fixed slot to memorize.
    position = ((entry.id or 0) + entry.cloze_index) % (len(options) + 1)
    options.insert(position, entry.translation or "")
    items = "".join(f"<li>{option}</li>" for option in options)
    sentence = f'<p><em>„{entry.sentence}"</em></p>' if entry.sentence else ""
    # Folded options: the learner tries to recall the answer before seeing
    # them, which beats plain multiple choice.
    front = (
        f"{sentence}"
        f"<p>Welche Übersetzung passt zu <strong>{entry.lemma}</strong>?</p>"
        f"<details><summary>Optionen zeigen</summary><ol>{items}</ol></details>"
    )
    return Exercise(front=front, back=_answer_line(entry))


def _word_hints(entry: Entry) -> list[str]:
    """Synonym and antonym lines that don't give the lemma away.

    Backstop for the prompt rule: `towards` would answer `toward`, and
    `undesirable` would answer `desirable`."""
    if not entry.lemma:
        return []
    lemma = entry.lemma.lower()
    hints = []
    for sign, word in (("≈", entry.synonym), ("≠", entry.antonym)):
        head = (word or "").split("(")[0].strip().lower()
        if head and lemma not in head and head not in lemma:
            hints.append(f"{sign} {word}")
    return hints


def _synonym_antonym(entry: Entry) -> Exercise:
    front = f"<p>{'<br>'.join(_word_hints(entry))}</p><p>{_WHICH_WORD}</p>"
    return Exercise(front=front, back=_word_back(entry))


def _definition(entry: Entry) -> Exercise:
    front = f"<p><em>{entry.definition_en}</em></p><p>{_WHICH_WORD}</p>"
    return Exercise(front=front, back=_word_back(entry))


def _listening(entry: Entry) -> Exercise:
    front = "<p>Was bedeutet das Wort, das du hörst?</p>"
    return Exercise(front=front, back=_word_back(entry), audio=True)


def _word_back(entry: Entry) -> str:
    ipa = f"<small>{entry.ipa}</small>" if entry.ipa else ""
    return f"<h2>{entry.lemma}</h2>{ipa}{_answer_line(entry)}"
