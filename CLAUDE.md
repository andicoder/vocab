# CLAUDE.md — vocab-api project conventions

This file is loaded automatically by Claude Code when working in this repo. It complements the user-level global CLAUDE.md and the company-wide rules; use both.

## Discipline

- **TDD.** Red → green → refactor. Add or change a failing test BEFORE adding production code. The exception is purely mechanical edits (rename, typo, formatting) and exploratory spikes that get reverted.
- **Clean-code.** Small functions, intention-revealing names, single responsibility. If a function reads top-down, no helpers needed; reach for extraction when the same chunk recurs or the body grows past ~25 LOC.
- **Function parameters: max 5.** Enforced via `ruff PLR0913`. Bundle related collaborators into a `@dataclass` (or pydantic model) and pass the bundle as a single param. Carve-out for FastAPI route handlers (per-file ignore in `pyproject.toml`) — `Depends()`-style injection is the framework's intended pattern. Other exceptions need an inline `# noqa: PLR0913` with a one-line reason.
- **Comments — light, at critical spots.** Trivial WHAT-comments (`# add the note`) and multi-paragraph docstrings stay forbidden. But brief comments are encouraged where the WHY is genuinely non-obvious from the code: third-party library quirks (anki Rust backend, edge-tts streaming, SQLAlchemy session lifecycle), race conditions, intentional ordering, transaction boundaries, workarounds with a reason. One short line is usually enough; two if needed. If a clearer name would remove the need for the comment, prefer the name.
- **Type hints everywhere.** `mypy --strict`-clean. Any `Any` is a deliberate decision; document it.
- **Lint-clean before commit.** `ruff check` + `ruff format` must pass.
- **PRs and commits are English.** Even when the conversation is in another language, PR titles, PR bodies, commit subjects and bodies are written in English.

## Stack

- Python 3.12+, FastAPI, SQLAlchemy 2.x async, asyncpg, Alembic, Pydantic v2 / pydantic-settings.
- src layout (`src/vocab_api/`). Editable install via `pip install -e ".[dev]"`.
- Tests: pytest (asyncio mode `auto`). Live in `tests/`.
- Image: `python:3.12-slim` base, single-stage build (see `Dockerfile`).

## Local workflow

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (hot path)
pytest -q

# Run app locally (needs reachable Postgres)
VOCAB_DATABASE_URL='postgresql+asyncpg://vocab:vocab@localhost:5432/vocab' \
  uvicorn vocab_api.main:app --reload

# Schema changes
alembic revision -m "what changed"          # write migration by hand (autogenerate
                                             # needs a live DB; we don't depend on it)
alembic upgrade head
```

## Conventions specific to this codebase

- **Auth boundary.** `auth.py::current_user` is the single trust point for `X-authentik-username`. Inside the app, pass `User` objects, never the raw header. The header arrives via Authentik forward-auth — production-only — so the test client sets it explicitly.
- **DB schema.** All ORM models live under the `vocab` Postgres schema (`Base.metadata = MetaData(schema=settings.db_schema)`). Migrations create `CREATE SCHEMA IF NOT EXISTS vocab` first.
- **Settings.** All runtime config goes through `vocab_api.config.Settings` (env-var prefix `VOCAB_`). Never read `os.environ` directly.
- **Async-first.** Every DB call goes through `AsyncSession`. No `def` route handlers in API code.
- **Migrations are hand-written.** Do not run `alembic revision --autogenerate` — it requires a live DB and we keep the dev path DB-free. Write the migration manually based on the model diff.

## Tests

- One file per module under `tests/test_<module>.py`.
- Use `httpx.ASGITransport` or FastAPI's `TestClient` for route tests.
- DB-touching tests get a transaction rollback per test (fixture). Don't wire in a separate test database — use the same Postgres + `BEGIN/ROLLBACK`.
- Coverage isn't enforced numerically; the rule is *every behavior change has an accompanying test*.

## Deployment

This repo only produces an image. The image is built + pushed to `ghcr.io/andicoder/vocab-api` by `.github/workflows/build.yml` on every push to `main` (tag `latest`) and on git tags `v*` (semver tag).

The deploy pipeline lives in a separate (private) infrastructure repo and consumes the image. This repo doesn't need to know about it.

## Out of scope (for now)

- Multi-language support beyond English source / German target. Schema is open for it but the code paths assume `en→de`.
