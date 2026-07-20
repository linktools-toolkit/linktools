#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EntrypointResolver: lists and resolves addressable objects inside extensions.
The first version implements ``list_entrypoints`` and ``resolve_agent`` only;
workflow/toolset resolution will land as separate Protocols when implemented.

Scoped agents are parsed from ``<root>/agents/<name>.md`` using the canonical
agent parser and re-id'd as ``extension:<id>:agent:<name>`` so the same entrypoint
name in two different extensions never collides in the global namespace."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..agent.spec import AgentSpec
from ..errors import ExtensionEntrypointNotFoundError, ExtensionNotFoundError

if TYPE_CHECKING:
    from .provider import DirectoryExtensionResourceProvider
from .entrypoint import EntrypointInfo, EntrypointListResult, EntrypointRef
from .resource import sanitize_extension_path
from .scope import ExtensionScope

_KIND_TO_DIR = {
    "agent": "agents",
    "skill": "skills",
    "tool": "tools",
    "mcp": "mcp",
    "workflow": "workflows",
    "script": "scripts",
}
_ENTRYPOINT_SUFFIX = {".md", ".yaml", ".yml", ".json", ".py"}
DEFAULT_ENTRYPOINT_LIMIT = 50


@runtime_checkable
class EntrypointResolver(Protocol):
    async def list_entrypoints(
        self,
        scope: ExtensionScope,
        *,
        kind: "str | None" = None,
        limit: int = DEFAULT_ENTRYPOINT_LIMIT,
        cursor: "str | None" = None,
    ) -> EntrypointListResult: ...

    async def resolve_agent(self, ref: EntrypointRef) -> AgentSpec: ...


class DirectoryEntrypointResolver:
    """EntrypointResolver over an ``extension_id -> root Path`` mapping."""

    def __init__(self, roots: "Mapping[str, Path | str]") -> None:
        self._roots: "dict[str, Path]" = {pid: Path(p) for pid, p in roots.items()}

    def _root_for(self, extension_id: str) -> "Path | None":
        return self._roots.get(extension_id)

    async def list_entrypoints(
        self,
        scope: ExtensionScope,
        *,
        kind: "str | None" = None,
        limit: int = DEFAULT_ENTRYPOINT_LIMIT,
        cursor: "str | None" = None,
    ) -> EntrypointListResult:
        root = self._root_for(scope.extension_id)
        if root is None:
            raise ExtensionNotFoundError(f"extension not found: {scope.extension_id}")
        kinds = (kind,) if kind else tuple(_KIND_TO_DIR.keys())
        entries: "list[EntrypointInfo]" = []
        for k in kinds:
            sub = root / _KIND_TO_DIR[k]
            if not sub.is_dir():
                continue
            for p in sorted(sub.iterdir()):
                if p.suffix not in _ENTRYPOINT_SUFFIX:
                    continue
                entries.append(
                    EntrypointInfo(kind=k, name=p.stem, extension_id=scope.extension_id)
                )
        offset = int(cursor) if cursor else 0
        page = entries[offset : offset + max(1, limit)]
        next_cursor = (
            str(offset + len(page)) if offset + len(page) < len(entries) else None
        )
        return EntrypointListResult(items=page, next_cursor=next_cursor)

    async def resolve_agent(self, ref: EntrypointRef) -> AgentSpec:
        if ref.kind != "agent":
            raise ExtensionEntrypointNotFoundError(
                f"resolve_agent only resolves kind='agent', got {ref.kind!r}"
            )
        scope = ref.scope
        if scope is None:
            raise ExtensionEntrypointNotFoundError("entrypoint ref has no extension scope")
        root = self._root_for(scope.extension_id)
        if root is None:
            raise ExtensionNotFoundError(f"extension not found: {scope.extension_id}")
        name = sanitize_extension_path(ref.name)
        candidate = (root / "agents" / name).with_suffix(".md")
        if not candidate.is_file():
            raise ExtensionEntrypointNotFoundError(
                f"extension agent not found: {scope.extension_id}/agents/{ref.name}"
            )
        # Reuse the canonical agent parser so scoped agents honor the same
        # frontmatter/tools/model contract as global agents.
        from ..catalog.parsing import parse_markdown_text
        from ..agent.codec import parse_agent_spec

        text = candidate.read_text(encoding="utf-8")
        payload, body = parse_markdown_text(text, source=str(candidate))
        scoped_id = f"extension:{scope.extension_id}:agent:{name}"
        agent = parse_agent_spec(scoped_id, payload, body)
        return agent


class ExtensionRegistry:
    """Default ExtensionSpecProvider AND ExtensionResourceProvider over a directory
    tree: each subdirectory of ``base_dir`` is an extension. The kind
    is read from an optional ``extension.yaml``; otherwise it defaults to 'custom'.
    Resource list/read delegate to a DirectoryExtensionResourceProvider so a single
    registry satisfies both Protocols."""

    def __init__(self, base_dir: "Path | str") -> None:
        self._base = Path(base_dir)
        # Lazily built; constructed on first resource access so ExtensionRegistry
        # stays cheap to instantiate when only spec listing is needed.
        self._resource_provider: "DirectoryExtensionResourceProvider | None" = None

    def _resources(self) -> "DirectoryExtensionResourceProvider":
        from .provider import DirectoryExtensionResourceProvider

        if self._resource_provider is None:
            roots = (
                {p.name: p for p in self._base.iterdir() if p.is_dir()}
                if self._base.is_dir()
                else {}
            )
            self._resource_provider = DirectoryExtensionResourceProvider(roots)
        return self._resource_provider

    async def list_ids(self) -> "tuple[str, ...]":
        if not self._base.is_dir():
            return ()
        return tuple(sorted(p.name for p in self._base.iterdir() if p.is_dir()))

    async def get(self, extension_id: str) -> Any:
        from .spec import ExtensionSpec

        root = self._base / extension_id
        if not root.is_dir():
            raise ExtensionNotFoundError(f"extension not found: {extension_id}")
        kind = "custom"
        name = extension_id
        manifest_yaml = root / "extension.yaml"
        if manifest_yaml.is_file():
            from ..catalog.parsing import parse_yaml_text

            payload = parse_yaml_text(
                manifest_yaml.read_text(encoding="utf-8"), source=str(manifest_yaml)
            )
            kind = str(payload.get("kind") or kind)
            name = str(payload.get("name") or name)
        return ExtensionSpec(id=extension_id, name=name, kind=kind)

    async def list_resources(
        self, scope, path: str = "", *, limit: int = 50, cursor: "str | None" = None
    ):
        return await self._resources().list_resources(
            scope, path, limit=limit, cursor=cursor
        )

    async def read_resource(self, ref, *, max_bytes: "int | None" = None):
        return await self._resources().read_resource(ref, max_bytes=max_bytes)


DirectoryExtensionRegistry = ExtensionRegistry
