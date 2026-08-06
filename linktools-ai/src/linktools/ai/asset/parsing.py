#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text parsers and filesystem/store asset loaders."""

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from ..core import JsonValue
from ..core.errors import AssetNotFoundError, AssetParseError
from .content import AssetContent, AssetContentInfo


class TextAssetStore(Protocol):
    async def get(self, path: str) -> "AssetContent | None": ...

    async def list_info(self) -> "tuple[AssetContentInfo, ...]": ...


class AssetLoader:
    def __init__(
        self,
        *,
        read: "Callable[[str], Awaitable[str]]",
        list_ids: "Callable[[str], Awaitable[tuple[str, ...]]]",
    ) -> None:
        self._read = read
        self._list_ids = list_ids

    @classmethod
    def from_filesystem(cls, *roots: Path) -> "AssetLoader":
        roots_tuple = tuple(roots)

        async def read(path: str) -> str:
            def read_sync() -> str | None:
                for root in roots_tuple:
                    candidate = root / path
                    if candidate.is_file():
                        return candidate.read_text(encoding="utf-8")
                return None

            value = await asyncio.to_thread(read_sync)
            if value is None:
                raise AssetNotFoundError(f"asset file not found: {path}")
            return value

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            def scan() -> "tuple[str, ...]":
                values: list[str] = []
                for root in roots_tuple:
                    if not root.is_dir():
                        continue
                    values.extend(
                        path.name[: -len(suffix)]
                        for path in sorted(root.iterdir())
                        if path.is_file() and path.name.endswith(suffix)
                    )
                return tuple(values)

            return await asyncio.to_thread(scan)

        return cls(read=read, list_ids=list_ids)

    @classmethod
    def from_store(cls, store: TextAssetStore, *, prefix: str) -> "AssetLoader":
        base = prefix.strip("/")

        def full(path: str) -> str:
            joined = f"{base}/{path.strip('/')}" if base else path.strip("/")
            if not joined or ".." in joined.split("/"):
                raise AssetNotFoundError(f"invalid asset path: {path!r}")
            return joined

        async def read(path: str) -> str:
            content = await store.get(full(path))
            if content is None:
                raise AssetNotFoundError(f"asset content not found: {path}")
            return content.content.decode("utf-8")

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            prefix_value = f"{base}/" if base else ""
            return tuple(
                sorted(
                    info.path.rsplit("/", 1)[-1][: -len(suffix)]
                    for info in await store.list_info()
                    if info.path.startswith(prefix_value) and info.path.endswith(suffix)
                )
            )

        return cls(read=read, list_ids=list_ids)

    async def read(self, path: str) -> str:
        return await self._read(path)

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._list_ids(suffix)

    def identity(self, raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _yaml_load(text: str, source: str) -> JsonValue:
    try:
        import yaml
    except ImportError as error:
        raise AssetParseError(f"{source}: YAML support requires PyYAML") from error
    try:
        value = yaml.safe_load(text)
    except Exception as error:
        raise AssetParseError(f"{source}: malformed YAML: {error}") from error
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AssetParseError(f"{source} must contain a YAML object")
    return value


def load_yaml_text(text: str, source: str = "<yaml>", *, resolve_env: bool = False) -> "dict[str, JsonValue]":
    data = _yaml_load(text, source)
    return _resolve_env_refs(data) if resolve_env else data


def load_markdown_text(text: str, source: str = "<md>") -> "tuple[dict[str, JsonValue], str]":
    if text.startswith("---\n"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return load_yaml_text(parts[1], source=source), parts[2]
    return {}, text


def parse_yaml_text(text: str, *, source: str = "<yaml>") -> "dict[str, JsonValue]":
    return load_yaml_text(text, source=source)


def parse_markdown_text(text: str, *, source: str = "<md>") -> "tuple[dict[str, JsonValue], str]":
    return load_markdown_text(text, source=source)


def parse_json_text(text: str, *, source: str = "<json>") -> "dict[str, JsonValue]":
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise AssetParseError(f"{source}: malformed JSON: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AssetParseError(f"{source}: JSON top-level must be an object")
    return value


def _resolve_env_refs(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _resolve_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    if not isinstance(value, str) or not value.startswith("env:"):
        return value
    for name in value[4:].split(":"):
        resolved = os.getenv(name)
        if resolved:
            return resolved
    return None


__all__ = [
    "AssetLoader",
    "TextAssetStore",
    "load_markdown_text",
    "load_yaml_text",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
]
