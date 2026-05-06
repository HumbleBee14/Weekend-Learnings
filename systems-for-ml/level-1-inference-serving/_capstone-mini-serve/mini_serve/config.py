"""Settings via Pydantic BaseSettings — twelve-factor config.

Reads from env vars (MINI_SERVE_*) with defaults. This is how production services do config:
no hardcoded values, no scattered globals, just one typed object.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINI_SERVE_", env_file=".env")

    # Model
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "auto"  # "float16", "bfloat16", "float32", or "auto"
    device: str = "auto"  # "cuda", "cpu", "mps", or "auto"

    # Batching
    max_batch_size: int = 8
    max_wait_ms: int = 10
    max_queue_size: int = 100  # backpressure: reject when queue exceeds this

    # Generation defaults (per-request can override)
    default_max_tokens: int = 128
    default_temperature: float = 0.7

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Observability
    log_level: str = "INFO"
    enable_metrics: bool = True


settings = Settings()
