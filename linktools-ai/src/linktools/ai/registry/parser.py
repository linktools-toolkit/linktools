#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared spec-loading primitives for the registry package: text parsers
(YAML/Markdown/JSON), a SpecLoader that reads from filesystem OR ResourceStore,
and helpers (parse_model_policy, parse_tool_refs) shared by the agent/swarm parsers."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from ..errors import InvalidSpecError, RegistryNotFoundError, RegistryParseError
from ..registry._config import load_markdown_text, load_yaml_text


def parse_yaml_text(text: str, *, source: str = "<yaml>") -> "dict[str, Any]":
    try:
        return load_yaml_text(text, source=source)
    except RegistryParseError:
        raise
    except Exception as exc:
        raise RegistryParseError(f"{source}: malformed YAML: {exc}") from exc


def parse_markdown_text(
    text: str, *, source: str = "<md>"
) -> "tuple[dict[str, Any], str]":
    try:
        return load_markdown_text(text, source)
    except Exception as exc:
        raise RegistryParseError(f"{source}: malformed Markdown: {exc}") from exc


def parse_json_text(text: str, *, source: str = "<json>") -> "dict[str, Any]":
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryParseError(f"{source}: malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryParseError(f"{source}: JSON top-level must be an object")
    return data


class SpecLoader:
    """Reads spec text + lists ids from either the filesystem or a ResourceStore."""

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
            raise RegistryNotFoundError(f"spec file not found: {path}")

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
            best = 0
            for root in roots_t:
                if root.is_dir():
                    for p in root.rglob("*"):
                        if p.is_file():
                            best = max(best, int(p.stat().st_mtime))
            return best

        return cls(read=read, list_ids=list_ids, revision=revision)

    @classmethod
    def from_resources(cls, resource_store: Any, *, prefix: str) -> "SpecLoader":
        # ResourceStore exposes get(ResourcePath) + propfind(ResourcePath); it has
        # no .list() and no global .revision(). Build paths via ResourcePath so the
        # store's own normalization + sandbox apply; pin revision at 0 (the store
        # owns per-resource etag/version, not a global revision clock).
        from ..storage.resource.models import Depth
        from ..storage.resource.path import ResourcePath

        base = prefix.strip("/")

        def _full(path: str) -> "ResourcePath":
            joined = f"{base}/{path.strip('/')}" if base else path.strip("/")
            if not joined or ".." in joined.split("/"):
                raise RegistryNotFoundError(f"invalid spec resource path: {path!r}")
            return ResourcePath(f"/{joined}")

        async def read(path: str) -> str:
            full = _full(path)
            resource = await resource_store.get(full)
            if resource is None:
                raise RegistryNotFoundError(f"spec resource not found: {full.value}")
            return resource.text()

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            root = ResourcePath(f"/{base}") if base else ResourcePath("/")
            page = await resource_store.propfind(root, depth=Depth.ONE, limit=1000)
            ids: "list[str]" = []
            for item in page.items:
                name = item.path.value.rstrip("/").rsplit("/", 1)[-1]
                if name.endswith(suffix):
                    ids.append(name[: -len(suffix)])
            return tuple(sorted(ids))

        async def revision() -> int:
            return 0

        return cls(read=read, list_ids=list_ids, revision=revision)

    async def read(self, path: str) -> str:
        return await self._read(path)

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return await self._list_ids(suffix)

    async def revision(self) -> int:
        return await self._revision()


def parse_model_policy(payload: "dict[str, Any]") -> Any:
    """Build a ModelPolicy from a YAML dict. Handles Decimal budget coercion."""
    from ..model.policy import ModelPolicy

    primary = payload.get("primary") or payload.get("model")
    if not primary:
        raise InvalidSpecError("model policy requires 'primary' (or 'model')")
    fallbacks = tuple(payload.get("fallbacks") or ())
    budget_raw = payload.get("budget")
    budget = Decimal(str(budget_raw)) if budget_raw is not None else None
    return ModelPolicy(
        primary=str(primary),
        fallbacks=fallbacks,
        max_retries=int(payload.get("max_retries", 1)),
        timeout_seconds=float(payload.get("timeout_seconds", 30.0)),
        max_tokens=payload.get("max_tokens"),
        budget=budget,
    )


def parse_tool_refs(items: Any) -> "tuple[Any, ...]":
    """Build a tuple[ToolRef] from a list of tool declarations.

    Accepted shapes:
      - "file"                 -> ToolRef(name="file")            (kind None -> builtin)
      - "builtin:file"         -> ToolRef(name="file", kind="builtin")
      - "skill:sql"            -> ToolRef(name="sql",  kind="skill")
      - {name: "file"}         -> ToolRef(name="file")
      - {kind: "skill", name: "sql", config: {...}} -> ToolRef(name, kind, config)
    """
    from ..agent.spec import ToolRef

    if items is None:
        # Distinguish "no tools key" (None -> runtime default) from "tools: []"
        # (empty tuple -> explicitly no tools) -- the three-state distinction.
        return None
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("tools must be a list")
    refs: list[Any] = []
    for item in items:
        if isinstance(item, str):
            refs.append(_tool_ref_from_string(item))
        elif isinstance(item, dict) and "name" in item:
            kind = item.get("kind")
            config = item.get("config") or {}
            if not isinstance(config, dict):
                raise InvalidSpecError(f"tool ref config must be a mapping: {item!r}")
            refs.append(
                ToolRef(
                    name=str(item["name"]),
                    kind=str(kind) if kind else None,
                    config=config,
                )
            )
        else:
            raise InvalidSpecError(f"invalid tool ref: {item!r}")
    return tuple(refs)


def _tool_ref_from_string(text: str) -> Any:
    """Split a 'kind:name' tool string; a bare name keeps kind None (resolver
    treats it as builtin) so prior ``tools: [file, terminal]`` is unchanged."""
    from ..agent.spec import ToolRef

    if ":" in text:
        kind, name = text.split(":", 1)
        kind = kind.strip()
        name = name.strip()
        if not kind or not name:
            raise InvalidSpecError(f"invalid tool ref: {text!r}")
        return ToolRef(name=name, kind=kind)
    name = text.strip()
    if not name:
        raise InvalidSpecError(f"invalid tool ref: {text!r}")
    return ToolRef(name=name)


class StrictConfigReader:
    """Strict, unknown-field-rejecting reader over a parsed config mapping (spec
    §13.1). Centralizes the primitive parsing every registry entity needs
    (bool / int / str / string-tuple / mapping) so each entity stops rolling its
    own ``_parse_bool`` / ``_validate_unknown``. Init rejects unknown keys."""

    def __init__(self, payload, *, allowed, context):
        self._payload = payload
        self._context = context
        unknown = sorted(set(payload) - set(allowed))
        if unknown:
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{context}: unknown fields: {', '.join(unknown)}")

    def required_str(self, name):
        value = self._payload[name]
        if not isinstance(value, str):
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a string")
        return value

    def optional_str(self, name):
        value = self._payload.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a string")
        return value

    def bool(self, name, default=None):
        if name not in self._payload:
            return default
        value = self._payload[name]
        if not isinstance(value, bool):
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a boolean")
        return value

    def non_negative_int(self, name, default=None):
        if name not in self._payload:
            return default
        value = self._payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            from ..errors import InvalidSpecError

            raise InvalidSpecError(
                f"{self._context}: {name} must be a non-negative integer"
            )
        return value

    def positive_number(self, name, default=None):
        if name not in self._payload:
            return default
        value = self._payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a positive number")
        return float(value)

    def string_tuple(self, name):
        value = self._payload.get(name)
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a list")
        return tuple(str(item) for item in value)

    def mapping(self, name):
        value = self._payload.get(name)
        if value is None:
            return None
        if not isinstance(value, dict):
            from ..errors import InvalidSpecError

            raise InvalidSpecError(f"{self._context}: {name} must be a mapping")
        return dict(value)
