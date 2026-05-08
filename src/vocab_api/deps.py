from fastapi import Request

from .anki_writer import AnkiWriter
from .audio import AudioStorage, TtsClient
from .gemini import GeminiClient


def get_gemini(request: Request) -> GeminiClient:
    return request.app.state.gemini  # type: ignore[no-any-return]


def get_tts(request: Request) -> TtsClient:
    return request.app.state.tts  # type: ignore[no-any-return]


def get_storage(request: Request) -> AudioStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_anki_writer(request: Request) -> AnkiWriter:
    return request.app.state.anki_writer  # type: ignore[no-any-return]
