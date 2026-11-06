"""Runtime configuration, read only from environment variables."""
from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # No defaults on purpose. The contract forbids hardcoded keys or endpoints,
    # and the scoring run points both of these at its own gateway.
    llm_base_url: str = ""
    llm_api_key: SecretStr = SecretStr("")

    database_url: SecretStr = SecretStr("")
    app_version: str = "dev"
    log_level: str = "INFO"

    # Must stay well under the compose healthcheck timeout, so a hung database
    # produces a clean 503 rather than a killed probe.
    health_db_timeout_seconds: float = 2.0

    # API
    max_upload_bytes: int = 50 * 1024 * 1024  # contract limit
    max_active_jobs: int = 10_000  # beyond this POST /jobs returns 429

    # Worker
    worker_concurrency: int = 5
    worker_poll_seconds: float = 0.5
    job_lease_seconds: float = 60.0  # a dead worker's job is retried after this long
    job_max_attempts: int = 3  # stops a document that kills workers from looping forever
    # Under the harness's 1800s limit, so a late job still reports why it failed.
    job_deadline_seconds: float = 1740.0
    sweep_interval_seconds: float = 15.0

    # LLM calls
    llm_request_timeout_seconds: float = 180.0  # one HTTP attempt
    llm_call_budget_seconds: float = 900.0  # one logical call, including all retries
    llm_max_concurrent: int = 20  # in-flight requests per worker process
    llm_backoff_base_seconds: float = 1.0
    llm_backoff_max_seconds: float = 60.0

    # Webhooks
    webhook_timeout_seconds: float = 10.0
    webhook_max_attempts: int = 8
    webhook_poll_seconds: float = 1.0
    webhook_backoff_base_seconds: float = 5.0
    # On because a local harness may use a private or host address. Link-local
    # addresses, which include cloud metadata endpoints, are always refused.
    webhook_allow_private: bool = True

    # Simulated work for the stub extractor, used to test load and crash recovery.
    stub_extractor_delay_seconds: float = 0.0

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url.strip() and self.llm_api_key.get_secret_value().strip())
