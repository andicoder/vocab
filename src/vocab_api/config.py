from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOCAB_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://vocab:vocab@localhost:5432/vocab"
    db_schema: str = "vocab"
    auth_user_header: str = "x-authentik-username"
    cors_origins: list[str] = []


settings = Settings()
