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


def parse_markdown_text(text: str, *, source: str = "<md>") -> "tuple[dict[str, Any], str]":
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
        async def read(path: str) -> str:
            file = await resource_store.get(f"{prefix}/{path}")
            if file is None:
                raise RegistryNotFoundError(f"spec resource not found: {prefix}/{path}")
            return file.content

        async def list_ids(suffix: str) -> "tuple[str, ...]":
            files = await resource_store.list(pattern=f"{prefix}/*{suffix}")
            ids: list[str] = []
            for f in files:
                name = f.path.rsplit("/", 1)[-1]
                ids.append(name[: -len(suffix)] if name.endswith(suffix) else name)
            return tuple(ids)

        async def revision() -> int:
            return await resource_store.revision()

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

    Accepted shapes (spec §10.1):
      - "file"                 -> ToolRef(name="file")            (kind None -> builtin)
      - "builtin:file"         -> ToolRef(name="file", kind="builtin")
      - "skill:sql"            -> ToolRef(name="sql",  kind="skill")
      - {name: "file"}         -> ToolRef(name="file")
      - {kind: "skill", name: "sql", config: {...}} -> ToolRef(name, kind, config)
    """
    from ..agent.spec import ToolRef

    if items is None:
        # Distinguish "no tools key" (None -> runtime default) from "tools: []"
        # (empty tuple -> explicitly no tools), per spec §10.7 three-state.
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
            refs.append(ToolRef(name=str(item["name"]), kind=str(kind) if kind else None,
                                config=config))
        else:
            raise InvalidSpecError(f"invalid tool ref: {item!r}")
    return tuple(refs)


def _tool_ref_from_string(text: str) -> Any:
    """Split a 'kind:name' tool string; a bare name keeps kind None (resolver
    treats it as builtin) so legacy ``tools: [file, terminal]`` is unchanged."""
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
