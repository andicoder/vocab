# vocab

Self-hosted vocab + translator + Anki-card-builder service for the family.

Collect words from any source (browser right-click, mobile share, Kindle import, manual) → translate with **Gemini Flash-Lite** (lemma, translation, alternatives, IPA, plausibility check, per-sense disambiguation) → generate audio via **edge-tts** → write finished **active-recall cards** (gap sentence on the front, word + audio + full sentence on the back) directly into our own **Anki sync server** → all Anki devices pick them up on the next normal sync.

[![CI](https://github.com/andicoder/vocab/actions/workflows/build.yml/badge.svg)](https://github.com/andicoder/vocab/actions/workflows/build.yml)

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Conventions: [`CLAUDE.md`](CLAUDE.md)

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

## Configuration

All runtime config flows through `vocab_api.config.Settings` (Pydantic) and is overridable via env vars with prefix `VOCAB_`. A `.env` file at repo root is also picked up. Never read `os.environ` directly in app code.

| Env var | Default | Purpose |
|---|---|---|
| `VOCAB_DATABASE_URL` | `postgresql+asyncpg://vocab:vocab@localhost:5432/vocab` | Async Postgres DSN. Required in any non-default deployment. |
| `VOCAB_DB_SCHEMA` | `vocab` | Postgres schema all ORM tables live under. |
| `VOCAB_AUTH_USER_HEADER` | `x-authentik-username` | Forward-auth header set upstream by Authentik. The app trusts this header — only expose the service behind the auth proxy. |
| `VOCAB_CORS_ORIGINS` | `[]` | Origins permitted by FastAPI's CORS middleware (browser extension). Comma-separated. |
| `VOCAB_LOG_LEVEL` | `INFO` | Root log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). See [`CLAUDE.md`](CLAUDE.md#logging) for level discipline. |
| `VOCAB_UI_DEFAULT_LOCALE` | `de` | Fallback UI locale when `Accept-Language` doesn't match a `locales/*.json`. |
| `VOCAB_PUBLIC_BASE_URL` | empty | Absolute base URL embedded in the bookmarklet (e.g. `https://vocab.example.com`). Falls back to `request.base_url` if empty. |
| **Gemini** | | |
| `VOCAB_GEMINI_API_KEY` | empty | Required for translation/plausibility. Empty disables the worker translation step (entries stay `pending`). |
| `VOCAB_GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model id. |
| `VOCAB_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | API base. Override for proxies/tests. |
| `VOCAB_GEMINI_TIMEOUT_S` | `10.0` | Per-request HTTP timeout. |
| **Audio (TTS + storage)** | | |
| `VOCAB_AUDIO_VOICE` | `en-US-AriaNeural` | edge-tts voice for synthesis. |
| `VOCAB_AUDIO_LOCAL_DIR` | `./var/audio` | Filesystem path for `LocalDirAudioStorage` (used when no S3 endpoint is configured). |
| `VOCAB_AUDIO_PUBLIC_URL_BASE` | empty | URL prefix returned to clients for stored audio (e.g. `/audio` or a CDN host). |
| `VOCAB_S3_ENDPOINT_URL` | empty | When set, audio is stored on S3-compatible object storage instead of disk. |
| `VOCAB_S3_REGION` | `fsn1` | S3 region. |
| `VOCAB_S3_BUCKET` | `vocab-media` | S3 bucket name. |
| `VOCAB_S3_ACCESS_KEY` | empty | S3 access key. Required when `VOCAB_S3_ENDPOINT_URL` is set. |
| `VOCAB_S3_SECRET_KEY` | empty | S3 secret key. |
| **Anki (file mode — dev/tests)** | | |
| `VOCAB_ANKI_COLLECTION_ROOT` | `./var/anki` | Filesystem root containing `<user>/collection.anki2`. Used when `VOCAB_ANKI_SYNC_URL` is empty. Conflicts with anki-sync-server in production (#5). |
| **Anki (sync mode — production)** | | |
| `VOCAB_ANKI_SYNC_URL` | empty | When set, the app uses `AnkiSyncWriter` over the Anki sync HTTP protocol against this server. |
| `VOCAB_ANKI_SYNC_CREDENTIALS_JSON` | `{}` | JSON object mapping vocab username → anki-sync-server password (loaded once at startup). Example: `{"alice":"pw1","bob":"pw2"}`. |
| `VOCAB_ANKI_SHADOW_ROOT` | `./var/anki-shadow` | Per-user "shadow" collection path the app keeps in sync with anki-sync-server. |

A minimal local `.env` for development:

```sh
VOCAB_DATABASE_URL=postgresql+asyncpg://vocab:vocab@localhost:5432/vocab
VOCAB_GEMINI_API_KEY=your-key-here
VOCAB_LOG_LEVEL=DEBUG
```

## Card design

Cards follow the active-recall pattern from Wozniak's *20 Rules of Formulating Knowledge*:

- **Front**: the source sentence with the target word replaced by `___`, plus a short German hint that disambiguates which word is being asked for. The user has to *produce* the English word from context — recognition cards alone build no productive vocabulary.
- **Back**: the word, audio (edge-tts), the full source sentence, alternative translations and IPA.

Polysemous words (`train` = railway / to exercise) land as **one card per meaning** — Gemini emits a stable `sense_key` slug so a second encounter under a different sense creates a second card, while a second encounter under the same sense is still dropped as a duplicate.

Each user picks a **card direction** via `PATCH /me/settings`:

| `card_direction` | Cards per note | Recall task |
|---|---|---|
| `de_en` (default) | one production card | see DE hint → recall EN word |
| `en_de` | one recognition card | see EN word → recall DE meaning |
| `both` | both | independent review schedules per direction |

New cards land in auto-named subdecks per `(source language, direction)` — e.g. `Englisch::DE→EN`, `Spanisch::EN→DE`, `Kroatisch::DE→EN`. Unknown ISO codes fall back to `UPPERCASE::DIRECTION`. Existing cards keep whichever deck they were created in.

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
