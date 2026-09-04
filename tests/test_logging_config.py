import json
import logging

import pytest

from vocab_api.logging_config import configure_logging

UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


@pytest.fixture(autouse=True)
def _restore_logging() -> object:
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    saved_uvicorn = {
        name: (logging.getLogger(name).handlers[:], logging.getLogger(name).propagate)
        for name in UVICORN_LOGGERS
    }
    yield None
    root.handlers, root.level = saved
    for name, (handlers, propagate) in saved_uvicorn.items():
        logging.getLogger(name).handlers = handlers
        logging.getLogger(name).propagate = propagate


def _emit(logger_name: str, message: str) -> str:
    logging.getLogger(logger_name).warning(message)
    return message


def test_json_format_emits_one_parseable_object_per_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json")
    _emit("vocab_api.routes", "hello")

    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {
        "time": pytest.approx(json.loads(line)["time"]),
        "level": "WARNING",
        "logger": "vocab_api.routes",
        "message": "hello",
    }


def test_the_field_names_match_the_rest_of_the_log_fleet() -> None:
    # Home Assistant, evcc, TeslaMate and OpenObserve are parsed into
    # time/level/logger/message by Fluent Bit. A producer that emits JSON with
    # its own names is structured but not queryable alongside them.
    from vocab_api.logging_config import JsonFormatter

    record = logging.LogRecord(
        name="vocab_api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    assert set(json.loads(JsonFormatter().format(record))) == {
        "time",
        "level",
        "logger",
        "message",
    }


def test_text_format_stays_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "text")
    _emit("vocab_api", "plain please")

    out = capsys.readouterr().out.strip()
    assert not out.startswith("{")
    assert "plain please" in out


def test_uvicorn_access_lines_go_through_our_formatter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The trap: uvicorn installs handlers on its own loggers before app code
    # runs, so its access lines keep uvicorn's format and bypass anything we
    # configure. Observed in the cluster as `INFO:     10.42.5.1:43218 - "GET
    # /healthz HTTP/1.1" 200 OK` sitting unparsed next to our own records.
    access = logging.getLogger("uvicorn.access")
    access.handlers = [logging.StreamHandler()]
    access.propagate = False

    configure_logging("INFO", "json")
    access.warning('127.0.0.1 - "GET /healthz HTTP/1.1" 200')

    line = capsys.readouterr().out.strip()
    assert json.loads(line)["logger"] == "uvicorn.access"


def test_a_traceback_survives_as_a_field(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("vocab_api").exception("failed")

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["message"] == "failed"
    assert "ValueError: boom" in payload["exception"]


def test_a_multiline_message_stays_one_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A bare newline in the message would split one record into two lines, and
    # the second is not valid JSON — the collector then files it as garbage.
    configure_logging("INFO", "json")
    _emit("vocab_api", "line one\nline two")

    out = capsys.readouterr().out.strip()
    assert len(out.splitlines()) == 1
    assert json.loads(out)["message"] == "line one\nline two"
