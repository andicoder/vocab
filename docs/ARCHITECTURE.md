# Architecture: vocab-api

Self-hosted vocab + translator + Anki-card-builder service for the family. This document describes the system as it is in code: components, data model, decisions. Implementation history lives in `git log`.

The infrastructure side (Authentik deployment, k8s manifests, ingress, DNS, Anki sync server) lives in a separate infrastructure repo. From the application's perspective, Authentik and the Anki sync server are **black boxes** — only their interfaces matter here.

---

## 1. Vision & scope

A self-hosted vocab/translation service for the whole family that …

- collects words from any source (browser right-click, mobile share, Kindle import, manual entry).
- translates each word with sentence context via **Gemini Flash-Lite** — including lemma, alternatives, IPA, plausibility check.
- generates **audio (TTS via `edge-tts`)** per word, MP3 in Hetzner Object Storage (or a local directory in dev).
- writes the finished cards directly into our **own Anki sync server in the cluster** — every Anki device picks them up on the next regular sync.
- is **multi-user from day one** (Authentik with one account per family member).
- is reachable from anywhere, secured behind SSO (**Authentik**).

**Architecture constraint:** the service runs **entirely in the cluster**. No workstation dependency. LLM = Gemini (external via API). Anki sync = the official Anki sync server (`python -m anki.syncserver`) in the cluster, replacing AnkiWeb.

### Out of scope

- Multiple learning languages (currently EN→DE only; the schema leaves room for more).
- Family decks / shared vocab pools.
- Custom web review UI as an Anki replacement (not needed — family members use AnkiMobile via Family Sharing).

### Consequences of switching to our own sync server

- **AnkiWeb browser review (`ankiweb.net/decks/`) goes away** — our own sync server has no web UI. Accepted.
- **All iOS devices need AnkiMobile** (€30 one-off per Apple Family Sharing group).
- **Every device** gets its custom sync URL switched to `anki-sync.example.com` — one-time setup step per device.

---

## 2. Architecture overview

```
┌─ Family members (anywhere) ──────────────────────────────────────────────┐
│                                                                          │
│  Brave/Firefox + WebExtension ──┐                                        │
│  Brave/Firefox + bookmarklet  ──┤                                        │
│  iOS Shortcut / Android share ──┼──▶ HTTPS                               │
│  Web UI (Quick-Add, Review)   ──┘   vocab.example.com                    │
│                                                                          │
│  AnkiMobile / Anki Desktop / AnkiDroid ──▶ HTTPS                         │
│                                            anki-sync.example.com         │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌─ k3s cluster ─────────────────────────────────────────────────────────────┐
│                                                                           │
│  nginx-ingress (forward-auth → Authentik) ── injects X-authentik-username │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─ vocab-api (this repo) ─────────────────────────────────────────┐     │
│  │  FastAPI + htmx UI + Gemini client + edge-tts                    │     │
│  │  endpoints: /, /queue, /bookmarklet, POST /vocab, /translate,    │     │
│  │             /audio/{word}.mp3, /import/kindle, /vocab/{id}/...   │     │
│  │  in-process worker: pending → translated → auto-approved/        │     │
│  │                     needs-review                                 │     │
│  │  multi-user via X-authentik-username                             │     │
│  └─┬───────────────────┬─────────────────┬─────────────────┬───────-┘     │
│    │                   │                 │                 │              │
│    ▼                   ▼                 ▼                 │              │
│  ┌─ Postgres ────┐  ┌─ Anki sync ──────┐  ┌─ S3 Hetzner ┐  │              │
│  │ schema:        │ │ python -m        │  │  bucket:    │  │              │
│  │ vocab          │ │ anki.syncserver  │  │  vocab-     │  │              │
│  │ multi-tenant   │ │ shared PVC       │  │  media      │  │              │
│  │ (user_id FK)   │ │ users from infra │  │  shared     │  │              │
│  │ + caches       │ │ vault            │  │  audio cache│  │              │
│  └────────────────┘ └──────────────────┘  └─────────────┘  │              │
│                                                            │              │
│  ┌─ Authentik (deployed by infra repo) ────┐               │              │
│  │  serves forward-auth headers            │               │              │
│  │  cookie domain: .example.com            │               │              │
│  └─────────────────────────────────────────┘               │              │
│                                                            │              │
└────────────────────────────────────────────────────────────┼──────────────┘
                                                             │ HTTPS
                                                             ▼
                                                  ┌─ Gemini API (external) ┐
                                                  │  translation + lemma   │
                                                  │  + alternatives + IPA  │
                                                  │  + plausibility check  │
                                                  └─────────────────────────┘
```

---

## 3. App components

### 3.1 vocab-api (FastAPI)

**Stack:** Python 3.12 + FastAPI + SQLAlchemy 2.x async + asyncpg + Jinja2 + htmx + edge-tts (no React/build step for the UI).

**Data model** (Postgres schema `vocab`, multi-user):

```sql
CREATE SCHEMA vocab;

CREATE TABLE vocab.user (
  id          BIGSERIAL PRIMARY KEY,
  username    TEXT NOT NULL UNIQUE,    -- from X-authentik-username
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE vocab.entry (
  id          BIGSERIAL PRIMARY KEY,
  user_id     BIGINT NOT NULL REFERENCES vocab.user(id),
  word        TEXT NOT NULL,           -- surface form from the source
  lemma       TEXT,                    -- normal form (LLM-normalized)
  sentence    TEXT,                    -- context sentence, optional
  translation TEXT,                    -- DE primary
  alternatives TEXT,                   -- DE alternatives, comma-separated
  ipa         TEXT,                    -- US-IPA in slashes
  audio_url   TEXT,                    -- public URL of the MP3
  source      TEXT,                    -- "Kindle: <book>" / URL / "manual"
  lang        TEXT NOT NULL DEFAULT 'en',
  status      TEXT NOT NULL DEFAULT 'pending',
              -- pending → translated → (auto-approved | needs-review) → synced
              -- (or rejected). Plausibility=YES → auto-approved; UNCLEAR → needs-review.
  anki_card_id BIGINT,                 -- card ID in the sync server
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  approved_at TIMESTAMPTZ,
  synced_at   TIMESTAMPTZ,
  meta        JSONB DEFAULT '{}'::jsonb,
  UNIQUE (user_id, lemma, lang)        -- per-user dedupe
);

CREATE INDEX idx_user_status ON vocab.entry(user_id, status);

-- Shared cache for translation + audio (cross-user)
CREATE TABLE vocab.translation_cache (
  id          BIGSERIAL PRIMARY KEY,
  word        TEXT NOT NULL,
  sentence_hash TEXT,                  -- sha256(sentence) or NULL for context-free
  lang        TEXT NOT NULL DEFAULT 'en',
  lemma       TEXT,
  translation TEXT,
  alternatives TEXT,
  ipa         TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (word, sentence_hash, lang)   -- NULLS NOT DISTINCT
);

CREATE TABLE vocab.audio_cache (
  id          BIGSERIAL PRIMARY KEY,
  word        TEXT NOT NULL,
  voice       TEXT NOT NULL DEFAULT 'en-US-AriaNeural',
  lang        TEXT NOT NULL DEFAULT 'en',
  s3_key      TEXT NOT NULL,           -- key in the storage backend
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (word, voice, lang)
);
```

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/vocab` | New word (UI/bookmarklet/extension/shortcut). User from header. |
| `GET`  | `/vocab` | Own list with `?status=...` filter. |
| `POST` | `/vocab/{id}/approve` | Finalize fields, write the card into the Anki sync server. |
| `POST` | `/vocab/{id}/reject` | Discard. |
| `POST` | `/translate` | On-demand translation without persisting (browser ext live popup). |
| `GET`  | `/audio/{word}.mp3` | Audio stream from storage (generates + caches if missing). |
| `POST` | `/import/kindle` | Multipart upload `vocab.db`. |
| `GET`  | `/` | Quick-Add UI. |
| `GET`  | `/queue` | Review UI + Kindle upload form. |
| `GET`  | `/bookmarklet` | Draggable bookmarklet for browsers. |

**Translation trigger:** try synchronously on `POST /vocab` (cache lookup first). On timeout → status `pending`, an in-process async background worker picks it up.

**Auto-approve flow:** after a successful translation call, run the plausibility check **immediately** (a second Gemini call passing the result):
- Plausibility **YES** → status `auto-approved` → write card into Anki sync server.
- Plausibility **UNCLEAR** or **NO** → status `needs-review` → waits in the review queue for a manual decision.

### 3.2 Web UI

- **`/` Quick-Add** — mobile-first, one input field plus optional sentence/source. Accepts `?word=&sentence=&source=` query params for prefill (used by the bookmarklet). PWA manifest, service worker for offline shell.
- **`/queue` Review** — shows **only `needs-review` cards** (plausibility UNCLEAR/NO). Inline-editable fields, Approve/Reject buttons. On approve: direct write into the Anki sync server. Auto-approved cards go to Anki immediately and never appear here.
- **i18n** — UI strings live in `locales/{de,en}.json` behind a tiny `t()` helper; `Accept-Language` picks the locale, fallback to `ui_default_locale` (de).

### 3.3 Browser extension

`extension/` ships an unpacked MV3 add-on for Brave/Chrome/Firefox. Two right-click entries on selected text:
- **vocab: Wort speichern** — POSTs the selection (plus ~80 chars of context and the page URL) to `/vocab`.
- **vocab: Übersetzung anzeigen** — calls `/translate` and shows a floating tooltip with translation, alternatives, IPA. A `+ vocab` button on the tooltip saves the entry.

Auth piggybacks on the Authentik cookie on `.example.com` (`credentials: "include"`).

### 3.4 Bookmarklet

`/bookmarklet` renders a draggable link whose `javascript:` URI opens `/?word=...&sentence=...&source=...` in a new tab. Avoids cross-origin/cookie hassles by deferring to the same-origin Quick-Add page (which is gated by Authentik forward-auth anyway).

### 3.5 Mobile

- **iOS:** AnkiMobile for review. Quick-Add additionally via:
  - iOS Shortcut "Save word" — takes selected/shared text → POST to `/vocab`. Available from any app's share sheet.
  - PWA of vocab-api (`/` as a homescreen app).
- **Android:** AnkiDroid + PWA share target.

### 3.6 Translator (Gemini Flash-Lite)

- **Provider:** Google Gemini API.
- **Model:** `gemini-2.5-flash-lite`.
- **Key:** API key arrives via vault secret as env var (`VOCAB_GEMINI_API_KEY`); nothing in the repo.
- **Call:** vocab-api hits `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent` directly.

**Single-call JSON output:**

```json
{
  "lemma": "expedition",
  "translation": "die Expedition",
  "alternatives": "der Forschungsausflug, die Reise",
  "ipa": "/ˌɛkspɪˈdɪʃən/"
}
```

**Plausibility check:** a second Gemini call passing the translation output. Reply YES/NO/UNCLEAR. UNCLEAR/NO → `needs-review`.

**Cache:** `translation_cache` table. Identical `(word, sentence_hash, lang)` pairs → no duplicate call. Cross-user.

**Throttle:** 1 req/s in the background worker, comfortably below free-tier limits.

### 3.7 Audio (TTS via edge-tts)

- **Engine:** [`edge-tts`](https://pypi.org/project/edge-tts/) — Python wrapper around Microsoft Edge TTS. Free, no API key, neural voices, MP3 output.
- **Voice (default):** `en-US-AriaNeural`.
- **Storage:** swappable behind an `AudioStorage` Protocol.
  - `S3AudioStorage` (production): Hetzner Object Storage bucket `vocab-media`, S3 credentials from vault.
  - `LocalDirAudioStorage` (dev/test): writes under `audio_local_dir`.
  - The factory picks based on whether `s3_endpoint_url` is set.
- **Key:** `sha256(word + voice + lang)[:16].mp3`.
- **Cache:** `audio_cache` table.
- **Delivery to Anki:** on approve, vocab-api copies the MP3 into the user's Anki media folder; the sync protocol propagates it on the next client sync.
- **Live audio in the browser:** `GET /audio/{word}.mp3` streams from local dir (FileResponse) or 302-redirects to S3.

### 3.8 Anki sync server integration (consumer side)

The sync server itself is infra (deployed via the infrastructure repo). vocab-api is a **write consumer** against the shared PVC.

**Write access:** the vocab-api pod mounts the `anki-sync-data` PVC read-write and uses the official `anki` Python package (`anki.collection.Collection`) to write directly into the relevant user's `collection.anki2`. The sync server reads on the next client sync and serves the new cards.

**Notetype "Vocab"** with fields `Word, Lemma, Sentence, Translation, Alternatives, IPA, Audio, Source, DateAdded` is created on first user setup (`anki_writer.py::_ensure_notetype`).

🤔 **Concurrency concern:** vocab-api and the sync server writing to the same SQLite DB at the same time. Anki uses SQLite in WAL mode → readers + 1 writer concurrently. vocab-api writes only a few cards per day; sync-server writes are short. Worst case: brief SQLite lock, retry with backoff. Not implemented yet — if it bites in production, switch to retry with backoff or talk to the sync server over HTTP instead.

### 3.9 Auth integration (consumer side)

**Authentik** runs in the cluster (infra repo). From the app's perspective, only three assumptions matter:
- nginx-ingress runs forward-auth against Authentik.
- On successful login, Authentik adds the header `X-authentik-username` to the upstream request.
- Cookie domain is `.example.com` → log in once per browser, valid across all subdomains.

App side: `auth.py::current_user` is the only place this header is read. After that, everything works with `User` objects. In tests the header is set explicitly; in the pod it comes from Authentik.

---

## 4. Repo layout

```
github.com/andicoder/vocab/
├── src/vocab_api/
│   ├── main.py             # FastAPI app + lifespan (worker, http client, storage, anki writer)
│   ├── config.py           # Settings (env-prefix VOCAB_)
│   ├── db.py               # SQLAlchemy engine (NullPool) + session factory
│   ├── models.py           # ORM: User, Entry, TranslationCache, AudioCache
│   ├── schemas.py          # Pydantic I/O
│   ├── auth.py             # X-authentik-username header parsing
│   ├── deps.py             # FastAPI dependency wrappers
│   ├── operations.py       # Approve/reject/import helpers shared by JSON + htmx routes
│   ├── worker.py           # process_entry pipeline + background worker loop
│   ├── gemini.py           # Gemini wrapper + translation cache
│   ├── audio.py            # edge-tts + storage protocol (S3 / local dir)
│   ├── anki_writer.py      # Per-user collection.anki2 write + media copy
│   ├── kindle.py           # vocab.db parser (latest lookup per word)
│   ├── i18n.py             # t(key, locale, **kwargs) + Accept-Language
│   ├── locales/            # de.json, en.json
│   ├── routes/             # FastAPI routers (vocab, translate, audio, ui, imports)
│   ├── templates/          # Jinja2 + htmx (base, index, queue, bookmarklet, partials)
│   └── static/             # PWA manifest + sw.js + icon + bookmarklet.js
├── extension/              # MV3 browser extension (manifest, background, content, options)
├── alembic/versions/       # hand-written migrations
├── tests/                  # pytest, asyncio mode auto, BEGIN/ROLLBACK + TRUNCATE fixture
├── docs/                   # this file
├── .github/workflows/      # build.yml (tests + image → ghcr.io/andicoder/vocab-api)
├── compose.yaml            # local Postgres for dev/test
├── Dockerfile              # python:3.12-slim, single-stage
├── pyproject.toml
├── alembic.ini
├── README.md
├── CLAUDE.md
└── LICENSE                 # MIT
```

Image tagging: `ghcr.io/andicoder/vocab-api:1.2.3` (semver), plus floating `:latest`. The infrastructure repo pins to a semver tag explicitly, never `:latest` in production.

---

## 5. Decisions made

| Topic | Decision |
|-------|----------|
| **Translator** | Gemini Flash-Lite (single JSON call + plausibility check) |
| **TTS** | edge-tts → S3 (Hetzner) or local dir, swappable behind a Protocol |
| **Anki sync** | official `python -m anki.syncserver` in-cluster, replaces AnkiWeb |
| **iOS review** | AnkiMobile (€30 one-off, Apple Family Sharing for everyone) |
| **Multi-user** | in the schema from day one, Authentik provides identity |
| **Repo layout** | app code here; image via ghcr.io; infrastructure repo deploys |
| **Browser extension** | from-scratch MV3, ~150 LOC. Cross-browser via `browser ?? chrome` alias |
| **Web framework** | htmx + Jinja2, no React/build step |
| **i18n** | JSON dictionaries + tiny `t()` helper, no Babel |
| **Postgres** | shared (`database` namespace, dedicated DB `vocab`) |
| **Test isolation** | TRUNCATE before each test + BEGIN/ROLLBACK on the test session; `NullPool` on the engine to avoid asyncpg cross-loop issues under TestClient |
| **Domains** | `auth.example.com`, `vocab.example.com`, `anki-sync.example.com` |

---

## 6. Risks / open items

- **Gemini free-tier change** — fallback Azure Translator (2M chars/month free permanently).
- **edge-tts is reverse-engineered** — fallback Google Cloud TTS or local Piper.
- **SQLite concurrency** between vocab-api and anki-sync-server (see §3.8) — mitigated via WAL + an HTTP bridge if needed.
- **Authentik forward-auth + WebExtension:** cookie-based auth requires `SameSite=None; Secure` for cross-origin calls from arbitrary websites.
- **AnkiWeb migration:** before the switch, sync each user once via Anki Desktop, then change the sync URL and do a full sync. Existing learning history is preserved.
- **Anki updates may change the sync protocol:** anki-sync-server has to be updated alongside. Pin the image to semver.
- **Authentik self-hosting lock-in:** if Authentik is down, every protected app is unreachable. Backup strategy + emergency bypass are documented in the infrastructure repo.

---

## 7. References

- Anki official sync server: https://docs.ankiweb.net/sync-server.html
- Anki Python package: https://pypi.org/project/anki/
- AnkiConnect API: https://foosoft.net/projects/anki-connect/
- Gemini API: https://ai.google.dev/api
- edge-tts: https://pypi.org/project/edge-tts/
- htmx: https://htmx.org/
- WebExtensions MV3: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Manifest_V3
- Apple Family Sharing for purchased apps: https://support.apple.com/en-us/HT201079
- Authentik: https://goauthentik.io/docs/
