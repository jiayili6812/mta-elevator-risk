"""Configuration loading and project-relative path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
FROZEN_TEST_CONFIG = PROJECT_ROOT / "config" / "frozen_test_period.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_CONFIG)


def project_path(relative_path: str | Path) -> Path:
    return PROJECT_ROOT / Path(relative_path)


def assert_frozen_test_config(config: dict[str, Any]) -> None:
    frozen = load_yaml(FROZEN_TEST_CONFIG)
    split = config["split"]
    if not frozen.get("locked"):
        raise RuntimeError("Frozen-test lock file is not marked as locked.")
    if split["frozen_test_start"] != frozen["frozen_test_start"]:
        raise RuntimeError("Pipeline frozen-test start differs from lock file.")
    if split["frozen_test_end"] != frozen["frozen_test_end"]:
        raise RuntimeError("Pipeline frozen-test end differs from lock file.")

