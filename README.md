# vocab

Self-hosted vocab + translator + Anki-card-builder service for the family.

Collect words from any source (browser right-click, mobile share, Kindle import, manual) → translate with **Gemini Flash-Lite** (lemma, translation, alternatives, IPA, plausibility check) → generate audio via **edge-tts** → write finished cards directly into our own **Anki sync server** → all Anki devices pick them up on the next normal sync.

## Status

| Phase | Content | Status |
|-------|---------|--------|
| 2c | API skeleton: FastAPI, auth header, Postgres schema, GHA → ghcr.io | ✅ done |
| 2d | Translator (Gemini) + audio (edge-tts) + plausibility + auto-approve | ✅ done |
| 2e | Web UI (htmx, PWA, bookmarklet) + write into the Anki sync server | ✅ done |
| 2f | Browser extension (MV3, from-scratch) — context menu + translation popup | ✅ done |
| 2g | Kindle importer (`vocab.db` upload, dedupe per user) | ✅ done |

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

## Browser extension

Lives in [`extension/`](extension/) as an unpacked MV3 add-on for Brave/Chrome and Firefox. It adds two right-click entries on selected text:

- **vocab: Wort speichern** — POSTs the selection (plus ~80 chars of context and the page URL) to `/vocab`.
- **vocab: Übersetzung anzeigen** — calls `/translate` and shows a floating tooltip with translation, alternatives and IPA. A `+ vocab` button on the tooltip saves the entry.

Auth piggybacks on the Authentik cookie on `.example.com`; the extension itself has no login.

### Install (Brave / Chrome / Edge)

1. Open `chrome://extensions/` (or `brave://extensions/`).
2. Toggle **Developer mode** on.
3. Click **Load unpacked** and pick the `extension/` directory.
4. Open the extension's options (puzzle-piece icon → ⋮ → *Options*) and set the API URL if it differs from `https://vocab.example.com`.

### Install (Firefox)

1. Open `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on…** and pick `extension/manifest.json`.
3. The add-on stays loaded until Firefox restarts. For permanent install, sign it through `about:addons` or run a self-signed unbranded/Developer Edition build.

### Verifying it works

Highlight a word on any page → right-click → *vocab: Wort speichern*. A native browser notification should confirm. Try *vocab: Übersetzung anzeigen* on the same selection to see the tooltip.

## Kindle import

Kindle e-readers keep your vocabulary lookups in `system/vocabulary/vocab.db` (a SQLite file). vocab-api can ingest it directly: every English word with a lookup lands in your queue as `pending`, with the most recent in-context sentence and the book title as the source. Already-imported words are skipped, so re-uploads after every reading session are safe.

### Get the file

1. Plug your Kindle in via USB. It mounts as a regular drive.
2. Copy `<Kindle>/system/vocabulary/vocab.db` somewhere convenient. The path is the same on PaperWhite, Oasis and base Kindles.

### Upload via the UI

Open `/queue`, scroll to the *Kindle import* section, pick the `vocab.db` and click **Hochladen**. The toast tells you how many words were new vs. already in your collection. New entries pick up Gemini translation + audio in the background worker (1 req/s).

### Upload via the API

```sh
curl -F "file=@vocab.db" -H "X-authentik-username: $USER" \
     https://vocab.example.com/import/kindle
# {"added": 42, "skipped": 7}
```

In production the Authentik forward-auth header replaces the explicit `X-authentik-username` (which is dev-only).

## Deployment

This repo only produces an image. `.github/workflows/build.yml` builds and pushes to `ghcr.io/andicoder/vocab-api` on every push to `main` (`:latest`) and on git tags `v*` (semver).

The deploy pipeline (k8s manifests, Authentik forward-auth, Anki sync server) lives in a separate infrastructure repo and consumes the image.

## License

MIT
