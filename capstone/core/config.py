from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    fieldops_chat_model: str = "openai:gpt-4.1-mini"
    fieldops_judge_model: str = "openai:gpt-4.1"
    fieldops_embedding_model: str = "text-embedding-3-small"
    fieldops_prompt_tag: str = "dev"
    fieldops_prompt_source: str = "local"  # local | langsmith
    fieldops_offline: bool = False  # deterministic classifier/responder, no LLM calls

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "fieldops-dev"

    database_url: str = ""
    redis_url: str = "redis://localhost:6380/0"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    nylas_api_key: str = ""
    nylas_grant_id: str = ""
    nylas_webhook_secret: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    fieldops_ml_tools: bool = False  # attach the MLflow models to the intake agent
    applicationinsights_connection_string: str = ""

    # API keys -> tenant ids, "key1:t1,key2:t2". In production this lives in a secrets store.
    fieldops_api_keys: str = "dev-key-t1:t1,dev-key-t2:t2,dev-key-t3:t3"

    @property
    def has_llm(self) -> bool:
        return not self.fieldops_offline and bool(self.openai_api_key or self.anthropic_api_key)

    def api_key_map(self) -> dict[str, str]:
        pairs = (p.split(":", 1) for p in self.fieldops_api_keys.split(",") if ":" in p)
        return {k.strip(): v.strip() for k, v in pairs}


@lru_cache
def get_settings() -> Settings:
    return Settings()
