import json
import shutil
import subprocess
from pathlib import Path

import pytest

_EXT = Path(__file__).resolve().parents[1] / "extension"


def test_manifest_is_valid_mv3():
    data = json.loads((_EXT / "manifest.json").read_text(encoding="utf-8"))
    assert data["manifest_version"] == 3
    assert data["name"] == "vocab"
    assert "contextMenus" in data["permissions"]
    assert "scripting" in data["permissions"]
    assert "storage" in data["permissions"]
    assert data["host_permissions"] == ["<all_urls>"]
    assert data["background"]["service_worker"] == "background.js"
    assert data["options_ui"]["page"] == "options.html"


def test_manifest_references_existing_files():
    data = json.loads((_EXT / "manifest.json").read_text(encoding="utf-8"))
    assert (_EXT / data["background"]["service_worker"]).is_file()
    assert (_EXT / data["options_ui"]["page"]).is_file()
    for icon in data["icons"].values():
        assert (_EXT / icon).is_file()
    for entry in data["content_scripts"]:
        for script in entry["js"]:
            assert (_EXT / script).is_file()


def test_manifest_registers_both_context_menu_actions():
    js = (_EXT / "background.js").read_text(encoding="utf-8")
    assert "vocab-save-selection" in js
    assert "vocab-translate-selection" in js
    assert "Wort speichern" in js
    assert "Übersetzung anzeigen" in js


def test_content_script_listens_for_show_translation():
    js = (_EXT / "content.js").read_text(encoding="utf-8")
    assert "show-translation" in js
    assert "runtime.onMessage" in js


def test_background_script_targets_vocab_endpoint():
    js = (_EXT / "background.js").read_text(encoding="utf-8")
    assert "/vocab" in js
    assert "/translate" in js
    assert "Authorization" in js
    assert "contextMenus" in js


def test_manifest_has_identity_permission():
    data = json.loads((_EXT / "manifest.json").read_text(encoding="utf-8"))
    assert "identity" in data["permissions"]


def test_options_page_has_oidc_fields():
    html = (_EXT / "options.html").read_text(encoding="utf-8")
    assert "oidcIssuer" in html
    assert "oidcClientId" in html


def test_options_js_implements_pkce_login():
    js = (_EXT / "options.js").read_text(encoding="utf-8")
    assert "launchWebAuthFlow" in js
    assert "code_challenge" in js
    assert "code_verifier" in js


def test_options_js_persists_oidc_settings():
    js = (_EXT / "options.js").read_text(encoding="utf-8")
    assert "oidcIssuer" in js
    assert "oidcClientId" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("filename", ["background.js", "content.js", "options.js"])
def test_extension_javascript_parses(filename: str):
    result = subprocess.run(
        ["node", "--check", str(_EXT / filename)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
