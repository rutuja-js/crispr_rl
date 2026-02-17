"""YAML config loader with profile merging support."""

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config file. Falls back to default config if path not given."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_profile_weights(config: dict, profile: str) -> dict[str, float]:
    """Extract weight dict for a given profile name."""
    profiles = config.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(profiles)}")
    return profiles[profile]


def get_constraints(config: dict) -> dict[str, Any]:
    """Return constraints sub-dict."""
    return config.get("constraints", {})


def get_rl_config(config: dict) -> dict[str, Any]:
    """Return RL hyper-parameter sub-dict."""
    return config.get("rl", {})
