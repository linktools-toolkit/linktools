#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit-root local Project discovery and identity."""

from dataclasses import dataclass
from pathlib import Path

from ..foundation.digest import sha256_digest
from .config import load_config


@dataclass(frozen=True, slots=True)
class LocalProject:
    root: Path
    config_path: "Path | None"
    config: "dict[str, object]"
    project_id: str

    @classmethod
    def discover(cls, start: "str | Path", *, root: "str | Path | None" = None) -> "LocalProject":
        """Discover the nearest project config, or use an explicit root."""
        if root is not None:
            candidate = Path(root).expanduser().resolve()
            config_path = candidate / ".linktools" / "config.yaml"
            return cls._build(candidate, config_path if config_path.exists() else None)
        current = Path(start).expanduser().resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            config_path = candidate / ".linktools" / "config.yaml"
            if config_path.exists():
                return cls._build(candidate, config_path)
        return cls._build(current, None)

    @classmethod
    def load(cls, root: "str | Path") -> "LocalProject":
        """Load a known root without searching parent directories."""
        candidate = Path(root).expanduser().resolve()
        config_path = candidate / ".linktools" / "config.yaml"
        return cls._build(candidate, config_path if config_path.exists() else None)

    @classmethod
    def _build(cls, root: Path, config_path: "Path | None") -> "LocalProject":
        config_bytes = config_path.read_bytes() if config_path is not None else b""
        project_id = sha256_digest((str(root).encode("utf-8") + b"\0" + config_bytes))
        return cls(root=root, config_path=config_path, config=load_config(config_path) if config_path else {}, project_id=project_id)


__all__ = ["LocalProject"]
