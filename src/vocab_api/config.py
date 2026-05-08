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


settings = Settings()
