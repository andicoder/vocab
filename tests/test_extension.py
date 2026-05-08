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
    assert data["host_permissions"] == ["https://vocab.example.com/*"]
    assert data["background"]["service_worker"] == "background.js"
    assert data["options_ui"]["page"] == "options.html"


def test_manifest_references_existing_files():
    data = json.loads((_EXT / "manifest.json").read_text(encoding="utf-8"))
    assert (_EXT / data["background"]["service_worker"]).is_file()
    assert (_EXT / data["options_ui"]["page"]).is_file()
    for icon in data["icons"].values():
        assert (_EXT / icon).is_file()


def test_background_script_targets_vocab_endpoint():
    js = (_EXT / "background.js").read_text(encoding="utf-8")
    assert "/vocab" in js
    assert 'credentials: "include"' in js
    assert "contextMenus" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("filename", ["background.js", "options.js"])
def test_extension_javascript_parses(filename: str):
    result = subprocess.run(
        ["node", "--check", str(_EXT / filename)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
