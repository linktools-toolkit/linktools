#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared text parsers and strict registry configuration helpers."""

import json
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml  # type: ignore

from ..errors import InvalidSpecError, SpecNotFoundError, SpecParseError


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


def load_yaml_text(
    text: str,
    source: str = "<yaml>",
    *,
    resolve_env: bool = False,
) -> dict[str, object]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a YAML object")
    return _resolve_env_refs(data) if resolve_env else data


def load_markdown_text(
    text: str,
    source: str = "<md>",
) -> tuple[dict[str, object], str]:
    if text.startswith("---\n"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return load_yaml_text(parts[1], source=source), parts[2]
    return {}, text


def parse_yaml_text(text: str, *, source: str = "<yaml>") -> "dict[str, Any]":
    try:
        return load_yaml_text(text, source=source)
    except SpecParseError:
        raise
    except Exception as exc:
        raise SpecParseError(f"{source}: malformed YAML: {exc}") from exc


def parse_markdown_text(
    text: str, *, source: str = "<md>"
) -> "tuple[dict[str, Any], str]":
    try:
        return load_markdown_text(text, source)
    except Exception as exc:
        raise SpecParseError(f"{source}: malformed Markdown: {exc}") from exc


def parse_json_text(text: str, *, source: str = "<json>") -> "dict[str, Any]":
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecParseError(f"{source}: malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecParseError(f"{source}: JSON top-level must be an object")
    return data


class SpecLoader:
    """Reads spec text and lists ids from a filesystem or feature source."""

    def __init__(self, *, read, list_ids, revision) -> None:
        self._read = read
        self._list_ids = list_ids
        self._revision = revision

    @classmethod
    def from_filesystem(cls, *roots: Path) -> "SpecLoader":
        roots_t = tuple(Path(r) for r in roots)

        async def read(path: str) -> str:
            for root in roots_t:
                candidate = root / path
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8")
            raise SpecNotFoundError(f"spec file not found: {path}")

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            ids: list[str] = []
            for root in roots_t:
                if not root.is_dir():
                    continue
                for p in sorted(root.iterdir()):
                    if p.is_file() and p.name.endswith(suffix):
                        ids.append(p.name[: -len(suffix)])
            return tuple(ids)

        async def revision() -> int:
            # High-resolution revision over the full file set: (relative path,
            # mtime_ns, size) per file. mtime_ns (nanosecond, not second-level)
            # avoids same-second collisions; hashing path+size too means an add or
            # delete changes the revision (the max-mtime approach missed those
            # and same-second modifies). A file that disappears between rglob
            # and stat is skipped -- the next revision reflects the final state.
            state: "list[tuple[str, int, int]]" = []
            for root in roots_t:
                if not root.is_dir():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    try:
                        stat = p.stat()
                    except FileNotFoundError:
                        continue
                    state.append(
                        (str(p.relative_to(root)), stat.st_mtime_ns, stat.st_size)
                    )
            return hash(tuple(sorted(state)))

        return cls(read=read, list_ids=list_ids, revision=revision)

    @classmethod
    def from_store(cls, store: Any, *, prefix: str) -> "SpecLoader":
        base = prefix.strip("/")

        def _full(path: str) -> str:
            joined = f"{base}/{path.strip('/')}" if base else path.strip("/")
            if not joined or ".." in joined.split("/"):
                raise SpecNotFoundError(f"invalid feature path: {path!r}")
            return joined

        async def read(path: str) -> str:
            document = await store.get(_full(path))
            if document is None:
                raise SpecNotFoundError(f"spec document not found: {path}")
            return document.content.decode("utf-8")

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            ids: "list[str]" = []
            for item in await store.list_info():
                if item.path.startswith(base + "/") and item.path.endswith(suffix):
                    ids.append(item.path.rsplit("/", 1)[-1][: -len(suffix)])
            return tuple(sorted(ids))

        async def revision() -> int | str:
            current = await store.current_revision()
            if current is not None:
                return current
            state = tuple(
                (item.path, item.version, item.etag, item.active)
                for item in await store.list_info()
            )
            return hash(state)

        return cls(read=read, list_ids=list_ids, revision=revision)

    async def read(self, path: str) -> str:
        return await self._read(path)

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._list_ids(suffix)

    async def revision(self) -> int:
        return await self._revision()


def resolved_name(reader: "StrictConfigReader", entity_id: str) -> str:
    """Resolve a spec's display name. A MISSING 'name' falls back to the entity
    id; an explicit empty/whitespace 'name' is a config error (it is present but
    blank, not a 'use the id' signal) and raises. The two must not be conflated
    by ``or entity_id`` -- that would silently turn ``name: ""`` into the id."""
    name = reader.optional_str("name")
    if name is None:
        return entity_id
    name = name.strip()
    if not name:
        raise InvalidSpecError(f"{reader.context}: 'name' must not be empty")
    return name


class StrictConfigReader:
    """Strict, unknown-field-rejecting reader over a parsed config mapping.

    Centralizes the primitive parsing every registry entity needs
    (bool / int / str / string-tuple / mapping) so each entity stops rolling its
    own ``_parse_bool`` / ``_validate_unknown``. Init rejects unknown keys. Each
    accessor distinguishes a MISSING field (returns its default) from an
    explicit ``null`` (raises InvalidSpecError) -- the two must not be
    conflated, or a typo'd ``field: null`` silently becomes "use the default".
    """

    def __init__(self, payload, *, allowed, context):
        self._payload = payload
        self._context = context
        unknown = sorted(set(payload) - set(allowed))
        if unknown:
            raise InvalidSpecError(f"{context}: unknown fields: {', '.join(unknown)}")

    def _present(self, name):
        """Return (present, value) for ``name``. ``present`` is False when the
        field is absent (the caller applies its default). An explicit ``null``
        is present-but-invalid and raises here so every accessor rejects it
        uniformly instead of treating it as missing."""
        if name not in self._payload:
            return False, None
        value = self._payload[name]
        if value is None:
            raise InvalidSpecError(f"{self._context}: {name} must not be null")
        return True, value

    def required_str(self, name):
        present, value = self._present(name)
        if not present:
            raise InvalidSpecError(f"{self._context}: {name} is required")
        if not isinstance(value, str):
            raise InvalidSpecError(f"{self._context}: {name} must be a string")
        return value

    def optional_str(self, name):
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, str):
            raise InvalidSpecError(f"{self._context}: {name} must be a string")
        return value

    @property
    def context(self) -> str:
        """The validation context label (used by shared helpers for errors)."""
        return self._context

    def bool(self, name, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, bool):
            raise InvalidSpecError(f"{self._context}: {name} must be a boolean")
        return value

    def non_negative_int(self, name, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise InvalidSpecError(
                f"{self._context}: {name} must be a non-negative integer"
            )
        return value

    def positive_number(self, name, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise InvalidSpecError(f"{self._context}: {name} must be a positive number")
        return float(value)

    def positive_int(self, name, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidSpecError(
                f"{self._context}: {name} must be a positive integer"
            )
        return value

    def non_negative_decimal(self, name):
        present, value = self._present(name)
        if not present:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise InvalidSpecError(f"{self._context}: {name} must be a number")
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise InvalidSpecError(
                f"{self._context}: {name} must be a valid number"
            ) from exc
        if not result.is_finite() or result < 0:
            raise InvalidSpecError(
                f"{self._context}: {name} must be finite and non-negative"
            )
        return result

    def string_mapping(self, name):
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, Mapping):
            raise InvalidSpecError(f"{self._context}: {name} must be a mapping")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
                raise InvalidSpecError(
                    f"{self._context}: {name} must be a string mapping"
                )
            result[key] = item
        return result

    def str_or_bool(self, name):
        present, value = self._present(name)
        if not present:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise InvalidSpecError(f"{self._context}: {name} must be a string or boolean")

    def enum(self, name, enum_type, *, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, str):
            raise InvalidSpecError(f"{self._context}: {name} must be a string")
        try:
            return enum_type(value)
        except ValueError as exc:
            raise InvalidSpecError(
                f"{self._context}: invalid {name}: {value!r}"
            ) from exc

    def string_tuple(self, name, *, default=None):
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, list):
            raise InvalidSpecError(f"{self._context}: {name} must be a list")
        result = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise InvalidSpecError(
                    f"{self._context}: {name}[{index}] must be a non-empty string"
                )
            result.append(item.strip())
        return tuple(result)

    def mapping(self, name):
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, Mapping):
            raise InvalidSpecError(f"{self._context}: {name} must be a mapping")
        return dict(value)
