import json
from collections.abc import Callable
from pathlib import Path

from fastapi import Request

from .config import settings

_LOCALES_DIR = Path(__file__).resolve().parent / "locales"
_CACHE: dict[str, dict[str, str]] = {}


def available_locales() -> list[str]:
    return sorted(p.stem for p in _LOCALES_DIR.glob("*.json"))


def _load(locale: str) -> dict[str, str]:
    if locale not in _CACHE:
        path = _LOCALES_DIR / f"{locale}.json"
        if path.is_file():
            _CACHE[locale] = json.loads(path.read_text(encoding="utf-8"))
        else:
            _CACHE[locale] = {}
    return _CACHE[locale]


def t(key: str, locale: str, /, **kwargs: object) -> str:
    template = _load(locale).get(key)
    if template is None and locale != settings.ui_default_locale:
        template = _load(settings.ui_default_locale).get(key)
    if template is None:
        return key
    return template.format(**kwargs) if kwargs else template


def translator_for(locale: str) -> Callable[..., str]:
    def _t(key: str, **kwargs: object) -> str:
        return t(key, locale, **kwargs)

    return _t


def current_locale(request: Request) -> str:
    available = set(available_locales())
    accept = request.headers.get("accept-language", "")
    for tag in accept.split(","):
        code = tag.split(";", 1)[0].strip().split("-", 1)[0].lower()
        if code in available:
            return code
    return settings.ui_default_locale
