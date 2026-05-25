"""Runtime settings loader with YAML + environment support."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from trading_engine.common.exceptions import ConfigurationError
from trading_engine.config.models import AppConfig


class EnvSettings(BaseSettings):
    """Environment overrides for infrastructure-critical fields."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TE_", extra="ignore")

    env: str = Field(default="dev")
    project_root: Path = Field(default=Path("."))

    data_raw_dir: Path | None = None
    data_processed_dir: Path | None = None
    data_cache_dir: Path | None = None
    duckdb_path: Path | None = None

    default_timezone: str | None = None
    default_resolution: str | None = None

    log_level: str | None = None
    log_json: bool | None = None


class RuntimeSettings(BaseModel):
    """Resolved runtime settings with absolute paths."""

    project_root: Path
    app: AppConfig

    @property
    def raw_dir(self) -> Path:
        return self.app.paths.data_raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.app.paths.data_processed_dir

    @property
    def cache_dir(self) -> Path:
        return self.app.paths.data_cache_dir

    @property
    def duckdb_path(self) -> Path:
        return self.app.paths.duckdb_path


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Base config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, dict):
        raise ConfigurationError("Config root must be a mapping")

    return content


def _resolve_path(project_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (project_root / maybe_relative).resolve()


def _apply_environment_overrides(config: AppConfig, env: EnvSettings, project_root: Path) -> AppConfig:
    config.environment = env.env

    if env.data_raw_dir is not None:
        config.paths.data_raw_dir = env.data_raw_dir
    if env.data_processed_dir is not None:
        config.paths.data_processed_dir = env.data_processed_dir
    if env.data_cache_dir is not None:
        config.paths.data_cache_dir = env.data_cache_dir
    if env.duckdb_path is not None:
        config.paths.duckdb_path = env.duckdb_path

    if env.default_timezone is not None:
        config.ingestion.default_timezone = env.default_timezone
    if env.default_resolution is not None:
        config.ingestion.default_resolution = env.default_resolution

    if env.log_level is not None:
        config.logging.level = env.log_level
    if env.log_json is not None:
        config.logging.json = env.log_json

    config.paths.data_raw_dir = _resolve_path(project_root, config.paths.data_raw_dir)
    config.paths.data_processed_dir = _resolve_path(project_root, config.paths.data_processed_dir)
    config.paths.data_cache_dir = _resolve_path(project_root, config.paths.data_cache_dir)
    config.paths.duckdb_path = _resolve_path(project_root, config.paths.duckdb_path)
    config.paths.log_dir = _resolve_path(project_root, config.paths.log_dir)

    return config


@lru_cache(maxsize=1)
def get_settings(config_file: str = "configs/base.yaml") -> RuntimeSettings:
    """Load and cache runtime settings."""
    env = EnvSettings()
    project_root = env.project_root.resolve()

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    yaml_config = _load_yaml_config(config_path)
    app = AppConfig.model_validate(yaml_config)
    app = _apply_environment_overrides(app, env, project_root)

    app.paths.data_raw_dir.mkdir(parents=True, exist_ok=True)
    app.paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    app.paths.data_cache_dir.mkdir(parents=True, exist_ok=True)
    app.paths.log_dir.mkdir(parents=True, exist_ok=True)
    app.paths.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    return RuntimeSettings(project_root=project_root, app=app)
