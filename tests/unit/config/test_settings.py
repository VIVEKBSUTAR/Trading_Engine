from __future__ import annotations

from pathlib import Path

from trading_engine.config.settings import get_settings


def test_get_settings_loads_yaml_and_resolves_paths(monkeypatch) -> None:
    project_root = Path.cwd()
    monkeypatch.setenv("TE_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("TE_ENV", "test")

    get_settings.cache_clear()
    settings = get_settings("configs/base.yaml")

    assert settings.app.environment == "test"
    assert settings.raw_dir.is_absolute()
    assert settings.processed_dir.is_absolute()
    assert settings.cache_dir.is_absolute()
    assert settings.duckdb_path.is_absolute()
