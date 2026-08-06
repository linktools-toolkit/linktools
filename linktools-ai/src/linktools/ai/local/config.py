#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local project configuration parsing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocalPolicy:
    """Bound local discovery and subagent execution resources."""

    max_skill_depth: int = 8
    max_concurrency: int = 4
    timeout_seconds: float = 300
    max_calls: int = 100

    def validate(self) -> None:
        if self.max_skill_depth < 0 or self.max_concurrency < 1 or self.timeout_seconds <= 0 or self.max_calls < 1:
            raise ValueError("local policy limits must be positive")


def load_config(path: Path) -> "dict[str, Any]":
    """Load a YAML config without scanning outside the explicit project."""
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


__all__ = ["LocalPolicy", "load_config"]
