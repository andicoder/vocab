import json
import logging
import sys
from typing import Any

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# uvicorn attaches handlers to these before any app code runs, so they have to
# be handed back to the root logger explicitly or their lines keep uvicorn's own
# format and never reach the formatter configured below.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """One JSON object per record, keyed the way the rest of the log fleet is."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str, log_format: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
