# Plan: vocab-api

App-side plan for the self-hosted vocab + translator + Anki-card-builder service. Mid-level detail — architecture, component boundaries, and sub-phase order are committed; per-file design happens iteratively in each PR.

Phase 1 (Kindle one-shot script) lives separately in the infrastructure repo and is shipped. This document describes **Phase 2** = the self-hosted service.

The infrastructure side (Authentik deployment, k8s manifests, ingress, DNS) lives in the infrastructure repo (`ansible-home`). From the application's perspective, Authentik and the Anki sync server are **black boxes** — only their interfaces matter here.

---

## 1. Vision & Scope

**Goal:** a self-hosted vocab/translation service for the whole family that …

- collects words from any source (browser right-click, mobile share, Kindle import, subtitles, podcasts, manual entry).
- translates each word with sentence context via **Gemini Flash-Lite** — including lemma, alternatives, IPA, plausibility check. Replaces Google Translate / DeepL for personal language learning.
- generates **audio (TTS via `edge-tts`)** per word, MP3 in Hetzner Object Storage.
- writes the finished cards directly into our **own Anki sync server in the cluster** — every Anki device picks them up on the next regular sync.
- is **multi-user from day one** (Authentik with one account per family member).
- is reachable from anywhere, secured behind SSO (**Authentik**).

**Architecture constraint:** the service runs **entirely in the cluster**. No workstation dependency. LLM = Gemini (external via API). Anki sync = the official Anki sync server (`python -m anki.syncserver`) in the cluster, replacing AnkiWeb.

### Out of scope (Phase 2)

- Migration of further existing services behind Authentik (separate plan in the infrastructure repo).
- Multiple learning languages (currently EN→DE only; the schema leaves room for more).
- Family decks / shared vocab pools (Phase 3, if needed).
- Custom web review UI as an Anki replacement (not needed — every family member uses AnkiMobile via Family Sharing).

### Consequences of switching to our own sync server

- **AnkiWeb browser review (`ankiweb.net/decks/`) goes away** — our own sync server has no web UI. Accepted.
- **All iOS devices need AnkiMobile** (€30 one-off per Apple Family Sharing group, ~€7.50 per family member with 4 people).
- **Every device** gets its custom sync URL switched to `anki-sync.example.com` — one-time setup step per device.
- **Existing AnkiWeb data** must be exported / synced before the switchover, then re-uploaded to the new server.

---

## 2. Architecture overview

```
┌─ Family members (Owner, partner, kid…) — anywhere ───────────────────┐
│                                                                          │
│  Brave/Firefox + WebExtension ──┐                                        │
│  Brave/Firefox + bookmarklet  ──┤                                        │
│  iOS Shortcut / Android share ──┼──▶ HTTPS                               │
│  Web UI (Quick-Add, Review)   ──┘   vocab.example.com                        │
│                                                                          │
│  AnkiMobile / Anki Desktop / AnkiDroid ──▶ HTTPS                         │
│                                            anki-sync.example.com             │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
┌─ k3s cluster ─────────────────────────────────────────────────────────────┐
│                                                                           │
│  nginx-ingress (forward-auth → Authentik) ── injects X-authentik-username │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─ vocab-api (this repo) ──────────────────────────────────────────┐    │
│  │  FastAPI + htmx UI + Gemini client + edge-tts                     │    │
│  │  endpoints: POST /vocab, /import/kindle, /vocab/{id}/...          │    │
│  │  htmx templates: /, /queue                                        │    │
│  │  multi-user via X-authentik-username                              │    │
│  └─┬───────────────────┬─────────────────┬─────────────────┬─────────┘    │
│    │                   │                 │                 │              │
│    ▼                   ▼                 ▼                 │              │
│  ┌─ Postgres ────┐  ┌─ Anki sync ──────┐  ┌─ S3 Hetzner ┐  │              │
│  │ schema:        │  │ python -m        │  │  bucket:     │  │              │
│  │ vocab          │  │ anki.syncserver  │  │  vocab-      │  │              │
│  │ multi-tenant   │  │ shared PVC       │  │  media       │  │              │
│  │ (user_id FK)   │  │ users from infra │  │  shared      │  │              │
│  │ + caches       │  │ vault            │  │  audio cache │  │              │
│  └────────────────┘  └──────────────────┘  └──────────────┘  │              │
│                                                              │              │
│  ┌─ Authentik (deployed by infra repo) ────┐                 │              │
│  │  serves forward-auth headers            │                 │              │
│  │  cookie domain: .example.com                │                 │              │
│  │  users: owner, partner, kid, …        │                 │              │
│  │  group: vocab                           │                 │              │
│  └─────────────────────────────────────────┘                 │              │
│                                                              │              │
└──────────────────────────────────────────────────────────────┼──────────────┘
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

**Data model** (Postgres, **multi-user**):

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
  audio_url   TEXT,                    -- S3 URL of the MP3
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
  UNIQUE (word, sentence_hash, lang)
);

CREATE TABLE vocab.audio_cache (
  id          BIGSERIAL PRIMARY KEY,
  word        TEXT NOT NULL,
  voice       TEXT NOT NULL DEFAULT 'en-US-AriaNeural',
  lang        TEXT NOT NULL DEFAULT 'en',
  s3_key      TEXT NOT NULL,           -- key in the Hetzner bucket
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (word, voice, lang)
);
```

User + Entry are already in place from 2c. translation_cache + audio_cache come with 2d.

**Endpoints (MVP):**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/vocab` | New word (UI/bookmarklet/extension/shortcut). User from header. |
| `POST` | `/import/kindle` | multipart upload `vocab.db` |
| `GET`  | `/vocab` | own list with `?status=...` filter |
| `POST` | `/vocab/{id}/translate` | trigger / retry LLM translation |
| `POST` | `/vocab/{id}/approve` | finalize fields, set status `approved`, write the card straight into the Anki sync server |
| `POST` | `/vocab/{id}/reject` | discard |
| `POST` | `/translate` | on-demand translation without persisting (for future browser live translation) |
| `GET`  | `/audio/{word}.mp3` | audio stream from S3 (for browser-extension live audio) |
| `GET`  | `/` | Quick-Add UI |
| `GET`  | `/queue` | review UI |

**Translation trigger:** try synchronously on `POST /vocab` (cache lookup first). On timeout → status `pending`, an in-process async background worker picks it up.

**Auto-approve flow (variant B):** after a successful translation call, run the plausibility check **immediately** (a second Gemini call passing the result):
- Plausibility **YES** → status `auto-approved` → write card into Anki sync server (with tag `auto`).
- Plausibility **UNCLEAR** or **NO** → status `needs-review` → waits in the review queue for a manual decision.

### 3.2 Web UI (two modes, one app)

- **`/` Quick-Add** — mobile-first, one input field plus optional sentence/source. PWA manifest, service worker for offline add.
- **`/queue` Review** — shows **only `needs-review` cards** (plausibility=UNCLEAR/NO). Inline-editable fields (translation, alternatives), buttons Approve/Reject/Re-translate. On approve: direct write into the Anki sync server. Auto-approved cards go to Anki immediately and never appear here — the queue is normally near-empty (~5% of words).

### 3.3 Browser extension

Strategy: **fork AnkiLingoFlash** (already exists, has the AI + AnkiConnect pattern), redirect the endpoint to vocab-api. Saves writing WebExtension boilerplate.

**Features:**
- **Right-click context menu** "Save vocab" on selected text → POST `/vocab` with `word=selection`, `sentence=DOM neighbourhood ~200 chars`, `source=tab URL`.
- **Double-click translation popup** (stretch): double-click on a word → small tooltip with live translation from `/translate` + audio player (`/audio/{word}.mp3`) + button "+ to vocab".

**Auth:** via cookie. The Authentik cookie is on `.example.com` → flows automatically (`credentials: include` in fetch).

**Distribution:** unpacked. Brave + Firefox both support that. Install once per family member.

**Location:** `extension/` (in this repo, separate build pipeline).

### 3.4 Mobile

- **iOS:** **AnkiMobile (€30 one-off, via Apple Family Sharing for up to 6 people)** for review. Quick-Add additionally via:
  - **iOS Shortcut** "Save word" — takes selected/shared text → POST. Available from any app's share sheet.
  - PWA of vocab-api (`/` as a homescreen app).
- **Android:** AnkiDroid (free) + PWA share target.

### 3.5 Translator (Gemini Flash-Lite)

- **Provider:** Google Gemini API.
- **Model:** `gemini-2.5-flash-lite` (4× cheaper than Flash, more than enough for vocab translation; proven in Phase 1).
- **Key:** API key arrives via vault secret as env var (`VOCAB_GEMINI_API_KEY`); nothing in the repo.
- **Call:** vocab-api hits `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent` directly from the pod.

**Single-call JSON output:** one Gemini call returns everything for the card:

```json
{
  "lemma": "expedition",
  "translation": "die Expedition",
  "alternatives": "der Forschungsausflug, die Reise",
  "ipa": "/ˌɛkspɪˈdɪʃən/"
}
```

**Plausibility check (belt-and-suspenders against "learning the wrong thing"):** a second Gemini call, passing the translation output: "Does this translation fit in context? Reply YES/NO/UNCLEAR." On `UNCLEAR` → status `needs-review`, the card waits for a manual review decision. Cost: <$0.0001 per word extra.

**Cache:** `translation_cache` table (see §3.1). Identical `(word, sentence_hash)` pairs → no duplicate call. Cross-user (family members benefit from each other).

**Throttle:** 1 req/s in the background worker, comfortably below free-tier limits.

**Risks:**
- **Free-tier change:** Gemini may tighten the free tier at some point — fallback would be switching to `gemini-2.5-flash` (~3× more expensive but still very cheap) or a provider switch (Azure Translator has 2M chars/month free permanently, same code path with a provider abstraction).
- **Privacy:** words + sentences go to Google. Acceptable for personal language learning.

### 3.6 Audio (TTS via edge-tts)

- **Engine:** [`edge-tts`](https://pypi.org/project/edge-tts/) — a Python wrapper around Microsoft Edge TTS. Free, no API key, neural voices, MP3 output. Proven in Phase 1.
- **Voice (default):** `en-US-AriaNeural`. Per-user selectable (setting in vocab-api).
- **Generation:** asynchronous on approve, in parallel with the Gemini translation.
- **Storage:** Hetzner Object Storage bucket `vocab-media`. S3 credentials via vault secret as env var (reused from the Nextcloud setup).
- **Cache:** `audio_cache` table + object-storage key `sha256(word + voice + lang)[:16].mp3`. Identical word + voice → one file, shared across users.
- **Delivery to Anki:** on approve, vocab-api copies the MP3 into the user's Anki media (in the sync-server PVC under `<user>/collection.media/<filename>.mp3`). The Anki sync protocol propagates media on the next client sync.
- **Live audio in the browser:** `GET /audio/{word}.mp3` → vocab-api streams from S3 (or generates + caches).

**Risks:**
- **edge-tts is a reverse-engineered API** — Microsoft can break it any time. Fallback: Google Cloud TTS, or local Piper in the cluster.

### 3.7 Anki sync server integration (consumer side)

The sync server itself is infra (deployed via the infrastructure repo). We interact as a **write consumer** against the shared PVC.

**Write access:** the vocab-api pod mounts the same `anki-sync-data` PVC read-write and uses the official `anki` Python package to write directly into the relevant user's `collection.anki21`. The sync server reads on the next client sync and serves the new cards.

🤔 **Concurrency concern:** vocab-api and the sync server writing to the same SQLite DB at the same time. Mitigations:
- Anki uses SQLite in WAL mode → readers + 1 writer concurrently.
- vocab-api writes only a few cards per day; sync-server writes are short.
- Worst case: brief SQLite lock, retry with backoff.
- If problematic: vocab-api talks to the sync server over HTTP (own sync login as a pseudo-client) instead of writing to the file directly.

**Notetype "Vocab"** with fields `Word, Lemma, Sentence, Translation, Alternatives, IPA, Audio, Source, DateAdded` is created on first user setup — same code as the Phase-1 script (`ensure_notetype()`).

**AnkiWeb migration (one-off per user):**
1. In Anki Desktop, sync once with AnkiWeb.
2. Switch sync URL to `https://anki-sync.example.com`.
3. Full sync (upload to new server).
4. iOS devices: install AnkiMobile (Family Sharing), set sync URL, do one full sync (download).
5. The AnkiWeb account is left at its last-sync state and not used afterwards.

### 3.8 Auth integration (consumer side)

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
│   ├── main.py             # FastAPI app
│   ├── config.py           # Settings
│   ├── db.py               # SQLAlchemy session
│   ├── models.py           # ORM models (User, Entry, …)
│   ├── schemas.py          # Pydantic I/O
│   ├── auth.py             # X-authentik-username header parsing
│   ├── routes/             # FastAPI routers
│   ├── gemini.py           # 2d: Gemini wrapper + cache
│   ├── audio.py            # 2d: edge-tts + S3 upload
│   ├── anki_writer.py      # 2e: writes to anki-sync-server collection.anki21
│   ├── kindle.py           # 2g: vocab.db parser (lifted from Phase-1 script)
│   └── templates/          # 2e: Jinja2: index.html, queue.html, partials/
├── alembic/versions/       # hand-written migrations
├── tests/                  # pytest, asyncio mode auto
├── extension/              # 2f: AnkiLingoFlash fork — MV3 manifest, content.js, …
├── docs/                   # this file lives here
├── .github/workflows/      # build.yml → ghcr.io/andicoder/vocab-api
├── Dockerfile              # python:3.12-slim, single-stage
├── pyproject.toml
├── alembic.ini
├── README.md
├── CLAUDE.md
└── LICENSE                 # MIT
```

Image tagging: `ghcr.io/andicoder/vocab-api:1.2.3` (semver), plus floating `:latest`. The infrastructure repo pins to a semver tag explicitly, never `:latest` in production.

---

## 5. Implementation in sub-phases

The order is chosen so that every step is independently testable. **2a + 2b** are pure infra (Authentik, Anki sync server) and were done in the infrastructure repo. **2c** was the API skeleton, ✅ shipped. From here on:

### 2c — API skeleton ✅ done

`POST /vocab`, `GET /vocab`, auth-header parsing, multi-user-isolated, DB init + Alembic migration, image build via GHA → ghcr.io. Status `pending` for every new entry.

### 2d — Translator (Gemini) + audio (edge-tts) — 🚧 in progress

Planned **bottom-up, in 3 PRs**, each independently testable:

**PR1 — Gemini client + translation_cache + plausibility (pure logic)**
1. `gemini.py`: async HTTP client, single JSON call → `TranslationResult(lemma, translation, alternatives, ipa)`. Tests via `httpx.MockTransport`.
2. Plausibility method: second call, output `YES|NO|UNCLEAR`.
3. Migration: `vocab.translation_cache` table + model.
4. Cache lookup wrapper around the Gemini client.

**PR2 — Audio (edge-tts) + S3 + audio_cache**
1. `audio.py`: edge-tts MP3 generation.
2. S3 upload (Hetzner, `vocab-media` bucket); S3 creds from settings.
3. Migration: `vocab.audio_cache` table + model.
4. Cache lookup wrapper around edge-tts.

**PR3 — Background worker + auto-approve state transitions**
1. In-process async task: scans `pending` entries, translates, plausibility-checks, generates audio, sets status.
2. Throttle 1 req/s against Gemini.
3. `POST /vocab` calls the translator synchronously (with cache); on timeout it falls through to the worker.
4. `POST /translate` (on-demand, no persistence) for future browser live translation.
5. `GET /audio/{word}.mp3` streams from S3.

**Success criteria (end of 2d):**
- A new word is translated + IPA + audio finished + auto-approved or needs-review within 5s.
- An identical word + sentence the second time comes from cache (no Gemini call).
- Plausibility distribution in real usage roughly 95% YES / 5% UNCLEAR.

### 2e — Web UI + write into Anki sync

1. Jinja2 templates: Quick-Add (`/`), review queue (`/queue`).
2. htmx endpoints for inline-edit, approve, reject.
3. PWA manifest + minimal service worker.
4. Bookmarklet provisioned.
5. **Approve flow:** write the card directly into the Anki sync server's `collection.anki21`, auto-create notetype "Vocab" if missing.
6. ✓ Success criterion: add a word from the phone via `/`, approve at the desktop under `/queue`, the card appears on iOS after the AnkiMobile sync.

### 2f — Browser extension

1. Fork AnkiLingoFlash into `extension/`.
2. Point the endpoint at `vocab.example.com/vocab`, auth via cookie.
3. Right-click context menu → POST.
4. (Stretch) double-click translation popup.
5. ✓ Success criterion: select a word on any web page, right-click, and it lands in the vocab service.

### 2g — Kindle importer

1. Port the Phase-1 script logic into `src/vocab_api/kindle.py`.
2. Endpoint `POST /import/kindle` (multipart upload).
3. UI: upload form on `/queue`.
4. ✓ Success criterion: upload `vocab.db` → all EN words land as `pending` in your own queue.

---

## 6. Decisions made

| Topic | Decision |
|-------|----------|
| **Translator** | Gemini Flash-Lite (single JSON call + plausibility check) |
| **TTS** | edge-tts → Hetzner S3 bucket `vocab-media` |
| **Anki sync** | official `python -m anki.syncserver` in-cluster, replaces AnkiWeb |
| **iOS review** | AnkiMobile (€30 one-off, Apple Family Sharing for everyone) |
| **Multi-user** | in the schema from day one, Authentik provides identity |
| **Repo layout** | app code here; image via ghcr.io; infrastructure repo deploys |
| **Browser extension** | fork AnkiLingoFlash instead of writing from scratch |
| **Postgres** | shared (`database` namespace, dedicated DB `vocab`) |
| **Domains** | `auth.example.com`, `vocab.example.com`, `anki-sync.example.com` |
| **Anki sync implementation** | Python package first (same image set as vocab-api), switch to Rust on perf issues |

---

## 7. Risks / thoughts

- **Gemini free-tier change** — fallback Azure Translator (2M chars/month free permanently).
- **edge-tts is reverse-engineered** — fallback Google Cloud TTS or local Piper.
- **SQLite concurrency** between vocab-api and anki-sync-server (see §3.7) — mitigated via WAL + an HTTP bridge if needed.
- **Authentik forward-auth + WebExtension:** cookie-based auth requires `SameSite=None; Secure` for cross-origin calls from arbitrary websites.
- **AnkiWeb migration:** before the switch, sync each user once via Anki Desktop, then change the sync URL and do a full sync. Existing learning history is preserved.
- **AnkiMobile €30:** accepted. Family Sharing covers all devices.
- **Anki updates may change the sync protocol:** anki-sync-server has to be updated alongside. Pin the image to semver.
- **Authentik self-hosting lock-in:** if Authentik is down, every protected app is unreachable. Backup strategy + emergency bypass are documented in the infrastructure repo.
- **Phase-1 Kindle script remains usable on its own.** If the Phase-2 service is unreachable, the old script can still push directly to AnkiConnect (local Anki).

---

## 8. References

- Anki official sync server: https://docs.ankiweb.net/sync-server.html
- Anki Python package: https://pypi.org/project/anki/
- AnkiConnect API: https://foosoft.net/projects/anki-connect/
- AnkiLingoFlash (fork base for the extension): https://github.com/pictoune/AnkiLingoFlash
- Gemini API: https://ai.google.dev/api
- edge-tts: https://pypi.org/project/edge-tts/
- htmx: https://htmx.org/
- WebExtensions MV3: https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Manifest_V3
- Apple Family Sharing for purchased apps: https://support.apple.com/en-us/HT201079
- Authentik: https://goauthentik.io/docs/
