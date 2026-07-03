#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions for YAML/Markdown parsing and agent-group config merging."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore

from .utils import resolve_ref


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_env_refs(item) for item in value)
    return resolve_ref(value)


def load_yaml_file(path: Path, *, resolve_env: bool = False) -> "dict[str, object]":
    text: str = path.read_text(encoding="utf-8")
    return load_yaml_text(text, source=str(path), resolve_env=resolve_env)


def load_yaml_text(text: str, source: str = "<yaml>", *, resolve_env: bool = False) -> "dict[str, object]":
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a YAML object")
    if resolve_env:
        data = _resolve_env_refs(data)
    return data


def load_markdown_text(text: str, source: str = "<md>") -> "tuple[dict[str, object], str]":
    """Parse Markdown text with optional YAML frontmatter."""
    if text.startswith("---\n"):
        splits = text.split("---", 2)
        if len(splits) == 3:
            return load_yaml_text(splits[1], source=source), splits[2]
    return {}, text


def load_markdown_file(path: Path) -> "tuple[dict[str, object], str]":
    """Parse a Markdown file with optional YAML frontmatter."""
    return load_markdown_text(path.read_text(encoding="utf-8"), source=str(path))


def as_str_dict(value: object) -> "dict[str, str]":
    """Coerce a dict-like value to dict[str, str], skipping falsy values."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if v}  # type: ignore[union-attr]
