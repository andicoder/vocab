# vocab

Self-hosted vocab + translator + Anki-card-builder service for the family.

Collect words from any source (browser right-click, mobile share, Kindle import, manual) → translate with **Gemini Flash-Lite** (lemma, translation, alternatives, IPA, plausibility check) → generate audio via **edge-tts** → write finished cards directly into our own **Anki sync server** → all Anki devices pick them up on the next normal sync.

## Status

| Phase | Content | Status |
|-------|---------|--------|
| 2c | API skeleton: FastAPI, auth header, Postgres schema, GHA → ghcr.io | ✅ done |
| 2d | Translator (Gemini) + audio (edge-tts) + plausibility + auto-approve | 🚧 in progress |
| 2e | Web UI (htmx) + direct write into the Anki sync server | planned |
| 2f | Browser extension (fork of AnkiLingoFlash) | planned |
| 2g | Kindle importer (port from the Phase-1 script) | planned |

Plan: [`docs/PLAN.md`](docs/PLAN.md) · Conventions: [`CLAUDE.md`](CLAUDE.md)

## Stack

Python 3.12 · FastAPI · SQLAlchemy 2.x async · asyncpg · Alembic · Pydantic v2 · htmx (web UI) · edge-tts · Gemini API.

## Quickstart

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Run the app locally (needs a reachable Postgres):

```sh
VOCAB_DATABASE_URL='postgresql+asyncpg://vocab:vocab@localhost:5432/vocab' \
  uvicorn vocab_api.main:app --reload
```

Schema changes:

```sh
alembic revision -m "what changed"   # hand-written, never autogenerate
alembic upgrade head
```

Full conventions (TDD, src layout, auth boundary, migration rules) in [`CLAUDE.md`](CLAUDE.md).

## Deployment

This repo only produces an image. `.github/workflows/build.yml` builds and pushes to `ghcr.io/andicoder/vocab-api` on every push to `main` (`:latest`) and on git tags `v*` (semver).

The deploy pipeline (k8s manifests, Authentik forward-auth, Anki sync server) lives in a separate infrastructure repo and consumes the image.

## License

MIT
