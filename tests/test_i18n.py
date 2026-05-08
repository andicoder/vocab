from unittest.mock import Mock

from vocab_api.i18n import available_locales, current_locale, t, translator_for


def test_available_locales_lists_de_and_en():
    assert "de" in available_locales()
    assert "en" in available_locales()


def test_t_returns_localised_string():
    assert t("nav.add", "de") == "Hinzufügen"
    assert t("nav.add", "en") == "Add"


def test_t_falls_back_to_default_locale_for_missing_key():
    assert t("nav.add", "fr") == "Hinzufügen"


def test_t_returns_key_when_completely_missing():
    assert t("does.not.exist", "de") == "does.not.exist"


def test_t_formats_kwargs():
    assert t("toast.added", "de", word="x", status="pending") == "x hinzugefügt — Status: pending"


def test_translator_for_binds_locale():
    de = translator_for("de")
    assert de("nav.add") == "Hinzufügen"
    assert de("toast.added", word="x", status="ok") == "x hinzugefügt — Status: ok"


def _request_with_accept(value: str | None) -> Mock:
    request = Mock()
    request.headers = {"accept-language": value} if value is not None else {}
    return request


def test_current_locale_picks_first_supported():
    assert current_locale(_request_with_accept("en-US,en;q=0.9,de;q=0.8")) == "en"


def test_current_locale_skips_unsupported_and_picks_match():
    assert current_locale(_request_with_accept("zh-CN,zh;q=0.9,de;q=0.8")) == "de"


def test_current_locale_falls_back_to_default():
    assert current_locale(_request_with_accept(None)) == "de"
    assert current_locale(_request_with_accept("zh,ja")) == "de"
