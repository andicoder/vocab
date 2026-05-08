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

    anki_collection_root: str = "./var/anki"
    anki_deck_name: str = "Default"

    ui_default_locale: str = "de"
    public_base_url: str = ""


settings = Settings()
