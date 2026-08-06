#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Text parsers, asset loaders, and strict configuration readers."""

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from ..foundation.errors import InvalidAssetError, AssetNotFoundError, AssetParseError

E = TypeVar("E", bound=Enum)


def _yaml_load(text: str, source: str) -> object:
    try:
        import yaml
    except ImportError as exc:
        raise AssetParseError(f"{source}: YAML support requires PyYAML") from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        raise AssetParseError(f"{source}: malformed YAML: {exc}") from exc


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_env_refs(item) for item in value)
    if not isinstance(value, str) or not value.startswith("env:"):
        return value
    for name in value[4:].split(":"):
        resolved = os.getenv(name)
        if resolved:
            return resolved
    return None


def load_yaml_text(text: str, source: str = "<yaml>", *, resolve_env: bool = False) -> "dict[str, object]":
    data = _yaml_load(text, source) or {}
    if not isinstance(data, dict):
        raise AssetParseError(f"{source} must contain a YAML object")
    return _resolve_env_refs(data) if resolve_env else data


def load_markdown_text(text: str, source: str = "<md>") -> "tuple[dict[str, object], str]":
    if text.startswith("---\n"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return load_yaml_text(parts[1], source=source), parts[2]
    return {}, text


def parse_yaml_text(text: str, *, source: str = "<yaml>") -> "dict[str, Any]":
    return load_yaml_text(text, source=source)


def parse_markdown_text(text: str, *, source: str = "<md>") -> "tuple[dict[str, Any], str]":
    try:
        return load_markdown_text(text, source)
    except AssetParseError:
        raise
    except Exception as exc:
        raise AssetParseError(f"{source}: malformed Markdown: {exc}") from exc


def parse_json_text(text: str, *, source: str = "<json>") -> "dict[str, Any]":
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssetParseError(f"{source}: malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AssetParseError(f"{source}: JSON top-level must be an object")
    return data


class AssetLoader:
    def __init__(self, *, read, list_ids) -> None:
        self._read = read
        self._list_ids = list_ids

    @classmethod
    def from_filesystem(cls, *roots: Path) -> "AssetLoader":
        roots_t = tuple(Path(root) for root in roots)

        async def read(path: str) -> str:
            def read_sync() -> "str | None":
                for root in roots_t:
                    candidate = root / path
                    if candidate.is_file():
                        return candidate.read_text(encoding="utf-8")
                return None

            text = await asyncio.to_thread(read_sync)
            if text is None:
                raise AssetNotFoundError(f"asset file not found: {path}")
            return text

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            def scan() -> "list[str]":
                ids: "list[str]" = []
                for root in roots_t:
                    if not root.is_dir():
                        continue
                    for path in sorted(root.iterdir()):
                        if path.is_file() and path.name.endswith(suffix):
                            ids.append(path.name[: -len(suffix)])
                return ids

            return tuple(await asyncio.to_thread(scan))

        return cls(read=read, list_ids=list_ids)

    @classmethod
    def from_store(cls, store: Any, *, prefix: str) -> "AssetLoader":
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
            ids = [
                item.path.rsplit("/", 1)[-1][: -len(suffix)]
                for item in await store.list_info()
                if item.path.startswith(prefix_value) and item.path.endswith(suffix)
            ]
            return tuple(sorted(ids))

        return cls(read=read, list_ids=list_ids)

    async def read(self, path: str) -> str:
        return await self._read(path)

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._list_ids(suffix)

    def identity(self, raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolved_name(reader: "StrictConfigReader", entity_id: str) -> str:
    name = reader.optional_str("name")
    if name is None:
        return entity_id
    name = name.strip()
    if not name:
        raise InvalidAssetError(f"{reader.context}: 'name' must not be empty")
    return name


class StrictConfigReader:
    def __init__(self, payload: Mapping[str, Any], *, allowed: "set[str] | tuple[str, ...]", context: str) -> None:
        self._payload = payload
        self._context = context
        unknown = sorted(set(payload) - set(allowed))
        if unknown:
            raise InvalidAssetError(f"{context}: unknown fields: {', '.join(unknown)}")

    @property
    def context(self) -> str:
        return self._context

    def _present(self, name: str) -> "tuple[bool, Any]":
        if name not in self._payload:
            return False, None
        value = self._payload[name]
        if value is None:
            raise InvalidAssetError(f"{self._context}: {name} must not be null")
        return True, value

    def required_str(self, name: str) -> str:
        present, value = self._present(name)
        if not present:
            raise InvalidAssetError(f"{self._context}: {name} is required")
        if not isinstance(value, str):
            raise InvalidAssetError(f"{self._context}: {name} must be a string")
        return value

    def optional_str(self, name: str) -> "str | None":
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, str):
            raise InvalidAssetError(f"{self._context}: {name} must be a string")
        return value

    def bool(self, name: str, default: "bool | None" = None) -> "bool | None":
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, bool):
            raise InvalidAssetError(f"{self._context}: {name} must be a boolean")
        return value

    def non_negative_int(self, name: str, default: "int | None" = None) -> "int | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAssetError(f"{self._context}: {name} must be a non-negative integer")
        return value

    def nullable_non_negative_int(self, name: str, default: "int | None" = None) -> "int | None":
        if name not in self._payload:
            return default
        value = self._payload[name]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidAssetError(f"{self._context}: {name} must be a non-negative integer or null")
        return value

    def positive_number(self, name: str, default: "float | None" = None) -> "float | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise InvalidAssetError(f"{self._context}: {name} must be a positive number")
        return float(value)

    def positive_int(self, name: str, default: "int | None" = None) -> "int | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidAssetError(f"{self._context}: {name} must be a positive integer")
        return value

    def non_negative_decimal(self, name: str) -> "Decimal | None":
        present, value = self._present(name)
        if not present:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise InvalidAssetError(f"{self._context}: {name} must be a number")
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise InvalidAssetError(f"{self._context}: {name} must be a valid number") from exc
        if not result.is_finite() or result < 0:
            raise InvalidAssetError(f"{self._context}: {name} must be finite and non-negative")
        return result

    def string_mapping(self, name: str) -> "dict[str, str] | None":
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, Mapping):
            raise InvalidAssetError(f"{self._context}: {name} must be a mapping")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
                raise InvalidAssetError(f"{self._context}: {name} must be a string mapping")
            result[key] = item
        return result

    def str_or_bool(self, name: str) -> "str | bool | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, (str, bool)):
            raise InvalidAssetError(f"{self._context}: {name} must be a string or boolean")
        raise InvalidAssetError(f"{self._context}: {name} must be a non-empty string or boolean")

    def enum(self, name: str, enum_type: type[E], default: "E | None" = None) -> "E | None":
        value = self.optional_str(name)
        if value is None:
            return default
        try:
            return enum_type(value)
        except ValueError as exc:
            choices = ", ".join(str(item.value) for item in enum_type)
            raise InvalidAssetError(f"{self._context}: {name} must be one of {choices}") from exc

    def string_tuple(self, name: str, *, default: "tuple[str, ...] | None" = None) -> "tuple[str, ...] | None":
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, list):
            raise InvalidAssetError(f"{self._context}: {name} must be a list")
        result = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise InvalidAssetError(f"{self._context}: {name}[{index}] must be a non-empty string")
            result.append(item.strip())
        return tuple(result)

    def mapping(self, name: str) -> "dict[str, Any] | None":
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, Mapping):
            raise InvalidAssetError(f"{self._context}: {name} must be an object")
        return dict(value)


__all__ = [
    "AssetLoader",
    "StrictConfigReader",
    "load_markdown_text",
    "load_yaml_text",
    "parse_json_text",
    "parse_markdown_text",
    "parse_yaml_text",
    "resolved_name",
]
