"""YAML configs with `_base_` inheritance."""

from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_config(path) -> dict:
    """Read a YAML config, resolving `_base_` (a path relative to the file)."""
    path = Path(path)
    with path.open() as file:
        config = yaml.safe_load(file) or {}
    base = config.pop("_base_", None)
    if base is not None:
        config = _merge(read_config(path.parent / base), config)
    return config


def load_config(path, module: str) -> dict:
    """`read_config`, checking that `checkpoint_dir` and the `module` section exist."""
    config = read_config(path)
    missing = {"checkpoint_dir", module}.difference(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config
