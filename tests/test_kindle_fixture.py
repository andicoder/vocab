import importlib.util
from pathlib import Path

from vocab_api.kindle import parse_kindle_vocab

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests" / "fixtures" / "kindle_vocab.db"


def _load_smoke_script():
    spec = importlib.util.spec_from_file_location(
        "smoke_translate", _ROOT / "scripts" / "smoke_translate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_db_has_a_sentence_for_every_smoke_target():
    smoke = _load_smoke_script()
    entries = {e.word: e for e in parse_kindle_vocab(_FIXTURE)}
    missing = [w for w in smoke.DEFAULT_TARGETS if not entries.get(w) or not entries[w].sentence]
    assert not missing


def test_example_db_carries_no_real_book_titles():
    titles = {e.source for e in parse_kindle_vocab(_FIXTURE)}
    assert titles and all(t.startswith("Book ") for t in titles)


def test_smoke_script_reads_the_example_db_by_default():
    smoke = _load_smoke_script()
    assert smoke._parse_args([]).kindle_db == _FIXTURE
