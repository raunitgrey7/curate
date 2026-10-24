"""Runtime settings read once from the environment.

Everything the service needs at runtime is a plain value here, so tests can construct a ``Settings``
directly and the serving layer never reads ``os.environ`` on the request path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Process-wide configuration.

    Attributes:
        data_dir: where datasets are cached (``CURATE_DATA_DIR``).
        artifact_dir: where the trained serving bundle lives (``CURATE_ARTIFACT_DIR``).
        api_key: when set, every endpoint except ``/health`` and ``/metrics`` requires the
            ``X-API-Key`` header (``APP_API_KEY``).
        seed: the default random seed (``CURATE_SEED``).
        llm_explanations: enable the optional LLM rewording of explanations (``CURATE_LLM_EXPLAIN``).
        llm_model: Anthropic model id used when LLM explanations are enabled (``CURATE_LLM_MODEL``).
        log_level: root log level (``CURATE_LOG_LEVEL``).
    """

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    artifact_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    api_key: str | None = None
    seed: int = DEFAULT_SEED
    llm_explanations: bool = False
    llm_model: str = "claude-opus-5-5"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables, falling back to repository defaults."""
        return cls(
            data_dir=_env_path("CURATE_DATA_DIR", PROJECT_ROOT / "data"),
            artifact_dir=_env_path("CURATE_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"),
            api_key=os.environ.get("APP_API_KEY") or None,
            seed=int(os.environ.get("CURATE_SEED", DEFAULT_SEED)),
            llm_explanations=_env_flag("CURATE_LLM_EXPLAIN"),
            llm_model=os.environ.get("CURATE_LLM_MODEL", "claude-opus-5-5"),
            log_level=os.environ.get("CURATE_LOG_LEVEL", "INFO").upper(),
        )
