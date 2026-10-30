"""Rendering for the rotating "Übung" Anki card.

Each entry carries up to four exercise payloads (multiple choice, synonym/
antonym, monolingual definition, production). The pool skips types whose
data is missing; `active_exercise` picks one via the entry's `cloze_index`
so the existing rotate-cloze job drives both the gap-sentence rotation and
the exercise-type rotation with a single counter.
"""

from dataclasses import dataclass

from .gemini import split_collocations
from .models import Entry


@dataclass(frozen=True, slots=True)
class Exercise:
    front: str
    back: str


def exercise_pool(entry: Entry) -> list[Exercise]:
    """All renderable exercises for `entry`, in stable order."""
    pool: list[Exercise] = []
    if entry.mc_distractors and entry.translation and entry.lemma:
        pool.append(_multiple_choice(entry))
    if entry.synonym and entry.lemma:
        pool.append(_synonym_antonym(entry))
    if entry.definition_en and entry.lemma:
        pool.append(_definition(entry))
    if entry.production_prompt and entry.production_solution:
        pool.append(_production(entry))
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
    front = (
        f"{sentence}"
        f"<p>Welche Übersetzung passt zu <strong>{entry.lemma}</strong>?</p>"
        f"<ol>{items}</ol>"
    )
    return Exercise(front=front, back=_answer_line(entry))


def _synonym_antonym(entry: Entry) -> Exercise:
    ask_antonym = " und ein Antonym" if entry.antonym else ""
    sentence = f'<p><small><em>„{entry.sentence}"</em></small></p>' if entry.sentence else ""
    front = f"<p>Nenne ein Synonym{ask_antonym} für <strong>{entry.lemma}</strong>.</p>{sentence}"
    back = f"<p>Synonym: <strong>{entry.synonym}</strong></p>"
    if entry.antonym:
        back += f"<p>Antonym: <strong>{entry.antonym}</strong></p>"
    return Exercise(front=front, back=back)


def _definition(entry: Entry) -> Exercise:
    front = f"<p><em>{entry.definition_en}</em></p><p>Welches englische Wort ist gemeint?</p>"
    ipa = f"<small>{entry.ipa}</small>" if entry.ipa else ""
    back = f"<h2>{entry.lemma}</h2>{ipa}{_answer_line(entry)}"
    return Exercise(front=front, back=back)


def _production(entry: Entry) -> Exercise:
    front = (
        f"<p>{entry.production_prompt}</p>"
        "<p>Bilde einen englischen Satz mit dem passenden Wort.</p>"
    )
    back = f"<p><em>{entry.production_solution}</em></p>{_answer_line(entry)}"
    return Exercise(front=front, back=back)
