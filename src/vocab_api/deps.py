from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .anki_writer import AnkiBackend
from .audio import AudioStorage, TtsClient
from .llm import LlmClient
from .worker import WorkerDeps


def get_worker_deps(request: Request) -> WorkerDeps:
    return request.app.state.worker_deps  # type: ignore[no-any-return]


# The getters below project a single field out of WorkerDeps so route
# handlers can keep narrow signatures and tests can override individual
# collaborators via FastAPI's dependency_overrides.
def get_llm(request: Request) -> LlmClient:
    return get_worker_deps(request).llm


def get_tts(request: Request) -> TtsClient:
    return get_worker_deps(request).tts


def get_storage(request: Request) -> AudioStorage:
    return get_worker_deps(request).storage


def get_anki_writer(request: Request) -> AnkiBackend:
    return get_worker_deps(request).anki_writer


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    return get_worker_deps(request).cache_session_factory
