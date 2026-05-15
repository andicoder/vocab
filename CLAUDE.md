# CLAUDE.md — vocab-api project conventions

This file is loaded automatically by Claude Code when working in this repo. It complements the user-level global CLAUDE.md and the company-wide rules; use both.

## Discipline

- **TDD.** Red → green → refactor. Add or change a failing test BEFORE adding production code. The exception is purely mechanical edits (rename, typo, formatting) and exploratory spikes that get reverted.
- **Clean-code.** Small functions, intention-revealing names, single responsibility. If a function reads top-down, no helpers needed; reach for extraction when the same chunk recurs or the body grows past the soft limit below.
- **Code size limits — all enforced via ruff.** Aim for the soft target; the hard ceiling is what fails CI.
  - **Line length: 100** (`line-length = 100`). PEP 8's 79 is too tight for modern editors; >120 makes side-by-side diffs unreadable.
  - **Function body: ~25 LOC soft, 30 statements hard** (`PLR0915`, `max-statements = 30`). Extract a helper when the body grows past that or when the same chunk recurs. Cohesive migration functions can earn a `# noqa: PLR0915` with a one-line reason — splitting them just to dodge the rule hides the logic.
  - **Function arguments: max 5** (`PLR0913`). Bundle related collaborators into a `@dataclass` or pydantic model and pass the bundle as a single param. Carve-out for FastAPI route handlers (per-file ignore in `pyproject.toml`) — `Depends()`-style injection is the framework's intended pattern. Other exceptions need an inline `# noqa: PLR0913` with a one-line reason.
  - **Cyclomatic complexity: max 10** (`C901`). Deeply branched logic is a refactor signal — usually a state machine, a polymorphism, or a guard-clause extraction.
  - **Statements per function, branches, return points: ruff defaults** (`PLR0912`, `PLR0911`). If you hit them you're almost certainly mixing concerns.
- **Comments — light, at critical spots.** Trivial WHAT-comments (`# add the note`) and multi-paragraph docstrings stay forbidden. But brief comments are encouraged where the WHY is genuinely non-obvious from the code: third-party library quirks (anki Rust backend, edge-tts streaming, SQLAlchemy session lifecycle), race conditions, intentional ordering, transaction boundaries, workarounds with a reason. One short line is usually enough; two if needed. If a clearer name would remove the need for the comment, prefer the name.
- **Type hints everywhere.** `mypy --strict`-clean. Any `Any` is a deliberate decision; document it.
- **Lint-clean before commit.** `ruff check` + `ruff format` must pass.
- **PRs, commits and issues are English.** Even when the conversation is in another language, issue titles, issue bodies, PR titles, PR bodies, commit subjects and commit bodies are written in English.

## Stack

- Python 3.12+, FastAPI, SQLAlchemy 2.x async, asyncpg, Alembic, Pydantic v2 / pydantic-settings.
- src layout (`src/vocab_api/`). Editable install via `pip install -e ".[dev]"`.
- Tests: pytest (asyncio mode `auto`). Live in `tests/`.
- Image: `python:3.12-slim` base, single-stage build (see `Dockerfile`).

## Local workflow

A `.venv` already exists at the project root. **Do not** `source .venv/bin/activate` — call the venv binaries by path so the shell stays clean.

```sh
# Initial setup (only when .venv is missing)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests (hot path) — Postgres must be reachable, see compose.yaml
.venv/bin/pytest -q

# Lint + format
.venv/bin/ruff check
.venv/bin/ruff format --check

# Run app locally (needs reachable Postgres)
VOCAB_DATABASE_URL='postgresql+asyncpg://vocab:vocab@localhost:5432/vocab' \
  .venv/bin/uvicorn vocab_api.main:app --reload

# Schema changes
alembic revision -m "what changed"          # write migration by hand (autogenerate
                                             # needs a live DB; we don't depend on it)
alembic upgrade head
```

## Issue / PR workflow

- One issue → one branch → one PR. Do not bundle multiple GitHub issues into one branch, even when the diffs would be small. Each PR references the issue it closes in the body (`Fixes #N`).
- Branch names: `fix/<slug>`, `feat/<slug>`, `chore/<slug>`. Slug is short, kebab-case, and reflects the change, not the ticket number.
- The test that proves the fix and the fix itself land in the same PR (TDD: red → green in the same branch, not split across PRs).
- Branch from `main`. Do not chain issue branches; rebase on `main` if it moves while the PR is open.

## Conventions specific to this codebase

- **Auth boundary.** `auth.py::current_user` is the single trust point for `X-authentik-username`. Inside the app, pass `User` objects, never the raw header. The header arrives via Authentik forward-auth — production-only — so the test client sets it explicitly.
- **DB schema.** All ORM models live under the `vocab` Postgres schema (`Base.metadata = MetaData(schema=settings.db_schema)`). Migrations create `CREATE SCHEMA IF NOT EXISTS vocab` first.
- **Settings.** All runtime config goes through `vocab_api.config.Settings` (env-var prefix `VOCAB_`). Never read `os.environ` directly.
- **Async-first.** Every DB call goes through `AsyncSession`. No `def` route handlers in API code.
- **Migrations are hand-written.** Do not run `alembic revision --autogenerate` — it requires a live DB and we keep the dev path DB-free. Write the migration manually based on the model diff.

## Logging

- **Setup.** `main.py` calls `logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)` once at module top-level. `force=True` because uvicorn installs handlers before our app code runs. Per-module loggers via `log = logging.getLogger(__name__)` — never `print` from production code.
- **Default level is `INFO`**, overridable per env via `VOCAB_LOG_LEVEL` (mapped through `Settings`, not `os.environ`). Common overrides: `DEBUG` for cache-hit traces, `WARNING` to silence routine outcomes.
- **Level discipline:**
  - **INFO** for terminal business outcomes (one per request/job): entry synced, duplicate dropped, import done.
  - **WARNING** for recoverable anomalies (timeout fallback, retry, missing optional config).
  - **ERROR** for aborts. Inside `except`, use `log.exception(...)` to include the stack trace.
  - **DEBUG** for verbose diagnostics off by default (cache hit/miss, request bodies).
- **Format.** `key=value` for structured-ish lines (`id=%s user=%s lemma=%s`). One line per outcome — no multi-line emit.
- **Don't log** secrets, raw passwords, or full API keys.
- **Alembic gotcha.** `alembic/env.py` calls `fileConfig(..., disable_existing_loggers=False)`. Without that flag, alembic's `[loggers]` block silences every non-listed logger after a migration runs — including `vocab_api.*` and any test fixture that captures records.

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
