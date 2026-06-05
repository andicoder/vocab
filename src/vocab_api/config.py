from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOCAB_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://vocab:vocab@localhost:5432/vocab"
    db_schema: str = "vocab"
    auth_user_header: str = "x-authentik-username"
    cors_origins: list[str] = []

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_s: float = 10.0

    s3_endpoint_url: str = ""
    s3_region: str = "fsn1"
    s3_bucket: str = "vocab-media"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    audio_voice: str = "en-US-AriaNeural"
    audio_local_dir: str = "./var/audio"
    audio_public_url_base: str = ""

    # File-based Anki write path (dev/tests). The pod mounts the
    # anki-sync-data PVC and writes directly into the user's collection.
    # Contended in production because anki-sync-server keeps the file open
    # at the Anki Rust backend level (#5).
    anki_collection_root: str = "./var/anki"

    # HTTP sync path (production). When `anki_sync_url` is set, the app
    # opens a private "shadow" collection under `anki_shadow_root` per user
    # and pushes notes to anki-sync-server via the official sync protocol.
    # `anki_sync_credentials_json` is a JSON object mapping vocab username
    # to the user's anki-sync-server password (loaded once at startup).
    anki_sync_url: str = ""
    anki_sync_credentials_json: str = "{}"
    anki_shadow_root: str = "./var/anki-shadow"

    ui_default_locale: str = "de"
    public_base_url: str = ""

    log_level: str = "INFO"

    # MCP server. HTTP transport always mounted at /mcp (Streamable HTTP,
    # stateless). When `mcp_api_key` is set, every tool call must supply it as
    # `x-api-key` or `Authorization: Bearer` — leave it empty only on trusted
    # private networks. Stdio transport: run `vocab-mcp` (no key needed).
    # `mcp_username` selects which vocab user the tools act on behalf of.
    mcp_api_key: str = ""
    mcp_username: str = "mcp"
    mcp_allowed_hosts: list[str] = []


settings = Settings()
