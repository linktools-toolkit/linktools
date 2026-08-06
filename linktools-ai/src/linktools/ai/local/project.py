#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit-root local Project discovery and identity."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import hashlib

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from linktools.core import ConfigField, ErrorProvider, PromptProvider

from ..core.json import JsonValue

OPENAI_BASE_URL = ConfigField.chain(
    PromptProvider("OpenAI Base URL", cached=True),
    ErrorProvider("no OpenAI base URL: pass --base-url, set OPENAI_BASE_URL, or run interactively to prompt"),
    name="OPENAI_BASE_URL",
    aliases=("base_url",),
    cast=str,
    required=True,
)
OPENAI_MODEL = ConfigField(
    name="OPENAI_MODEL",
    aliases=("model",),
    cast=str,
    provider=PromptProvider("Default model", cached=True),
)
OPENAI_API_KEY = ConfigField.chain(
    PromptProvider("OpenAI API Key", password=True, cached=False),
    ErrorProvider("no OpenAI API key: pass --api-key or set OPENAI_API_KEY"),
    name="OPENAI_API_KEY",
    aliases=("api_key",),
    cast=str,
    required=True,
    secret=True,
)


@dataclass(frozen=True, slots=True)
class LocalPolicy:
    max_skill_depth: int = 8
    max_concurrency: int = 4
    timeout_seconds: float = 300
    max_calls: int = 100

    def validate(self) -> None:
        if self.max_skill_depth < 0 or self.max_concurrency < 1 or self.timeout_seconds <= 0 or self.max_calls < 1:
            raise ValueError("local policy limits must be positive")


@dataclass(frozen=True, slots=True)
class LocalProject:
    root: Path
    storage_root: Path
    config_path: "Path | None"
    config: "dict[str, JsonValue]"
    project_id: str

    @classmethod
    def discover(
        cls,
        start: "str | Path",
        *,
        root: "str | Path | None" = None,
        storage_root: "str | Path | None" = None,
    ) -> "LocalProject":
        """Discover the work root and optionally use a separate storage root."""
        if root is not None:
            candidate = Path(root).expanduser().resolve()
            config_path = candidate / ".linktools" / "config.yaml"
            return cls._build(candidate, config_path if config_path.exists() else None, storage_root)
        current = Path(start).expanduser().resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            config_path = candidate / ".linktools" / "config.yaml"
            if config_path.exists():
                return cls._build(candidate, config_path, storage_root)
        return cls._build(current, None, storage_root)

    @classmethod
    def load(cls, root: "str | Path", *, storage_root: "str | Path | None" = None) -> "LocalProject":
        """Load a known work root without searching parent directories."""
        candidate = Path(root).expanduser().resolve()
        config_path = candidate / ".linktools" / "config.yaml"
        return cls._build(candidate, config_path if config_path.exists() else None, storage_root)

    @classmethod
    def _build(cls, root: Path, config_path: "Path | None", storage_root: "str | Path | None") -> "LocalProject":
        config_bytes = config_path.read_bytes() if config_path is not None else b""
        project_id = hashlib.sha256(str(root).encode("utf-8") + b"\0" + config_bytes).hexdigest()
        resolved_storage_root = root if storage_root is None else Path(storage_root).expanduser().resolve()
        return cls(
            root=root,
            storage_root=resolved_storage_root,
            config_path=config_path,
            config=load_config(config_path) if config_path else {},
            project_id=project_id,
        )


def load_config(path: Path) -> 'dict[str, JsonValue]':
    """Load only the configuration file belonging to the discovered project."""
    if not path.exists():
        return {}
    if _yaml is None:
        return {}
    value = _yaml.safe_load(path.read_text(encoding="utf-8"))
    return cast(dict[str, JsonValue], value) if isinstance(value, dict) else {}


__all__ = ["LocalPolicy", "LocalProject", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
