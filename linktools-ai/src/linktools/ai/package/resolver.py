#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EntrypointResolver: lists and resolves addressable objects inside packages
(spec §13.8). The first version implements ``list_entrypoints`` and
``resolve_agent``; ``resolve_toolset`` / ``resolve_workflow`` are reserved
(NotImplementedError) per spec §13.8.

Scoped agents are parsed from ``<root>/agents/<name>.md`` using the canonical
agent parser and re-id'd as ``package:<id>:agent:<name>`` so the same entrypoint
name in two different packages never collides in the global namespace."""

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ..agent.spec import AgentSpec
from ..errors import PackageEntrypointNotFoundError, PackageNotFoundError
from .entrypoint import EntrypointInfo, EntrypointListResult, EntrypointRef
from .resource import sanitize_package_path
from .scope import PackageScope

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
        scope: PackageScope,
        *,
        kind: "str | None" = None,
        limit: int = DEFAULT_ENTRYPOINT_LIMIT,
        cursor: "str | None" = None,
    ) -> EntrypointListResult:
        ...

    async def resolve_agent(self, ref: EntrypointRef) -> AgentSpec:
        ...


class DirectoryEntrypointResolver:
    """EntrypointResolver over a ``package_id -> root Path`` mapping."""

    def __init__(self, roots: "Mapping[str, Path | str]") -> None:
        self._roots: "dict[str, Path]" = {pid: Path(p) for pid, p in roots.items()}

    def _root_for(self, package_id: str) -> "Path | None":
        return self._roots.get(package_id)

    async def list_entrypoints(
        self,
        scope: PackageScope,
        *,
        kind: "str | None" = None,
        limit: int = DEFAULT_ENTRYPOINT_LIMIT,
        cursor: "str | None" = None,
    ) -> EntrypointListResult:
        root = self._root_for(scope.package_id)
        if root is None:
            raise PackageNotFoundError(f"package not found: {scope.package_id}")
        kinds = (kind,) if kind else tuple(_KIND_TO_DIR.keys())
        entries: "list[EntrypointInfo]" = []
        for k in kinds:
            sub = root / _KIND_TO_DIR[k]
            if not sub.is_dir():
                continue
            for p in sorted(sub.iterdir()):
                if p.suffix not in _ENTRYPOINT_SUFFIX:
                    continue
                entries.append(EntrypointInfo(
                    kind=k, name=p.stem, package_id=scope.package_id))
        offset = int(cursor) if cursor else 0
        page = entries[offset:offset + max(1, limit)]
        next_cursor = str(offset + len(page)) if offset + len(page) < len(entries) else None
        return EntrypointListResult(items=page, next_cursor=next_cursor)

    async def resolve_agent(self, ref: EntrypointRef) -> AgentSpec:
        if ref.kind != "agent":
            raise PackageEntrypointNotFoundError(
                f"resolve_agent only resolves kind='agent', got {ref.kind!r}"
            )
        scope = ref.scope
        if scope is None:
            raise PackageEntrypointNotFoundError("entrypoint ref has no package scope")
        root = self._root_for(scope.package_id)
        if root is None:
            raise PackageNotFoundError(f"package not found: {scope.package_id}")
        name = sanitize_package_path(ref.name)
        candidate = (root / "agents" / name).with_suffix(".md")
        if not candidate.is_file():
            raise PackageEntrypointNotFoundError(
                f"package agent not found: {scope.package_id}/agents/{ref.name}"
            )
        # Reuse the canonical agent parser so scoped agents honor the same
        # frontmatter/tools/model contract as global agents.
        from ..registry.parser import parse_markdown_text
        from ..registry.agent import parse_agent_spec

        text = candidate.read_text(encoding="utf-8")
        payload, body = parse_markdown_text(text, source=str(candidate))
        scoped_id = f"package:{scope.package_id}:agent:{name}"
        agent = parse_agent_spec(scoped_id, payload, body)
        return agent

    async def resolve_toolset(self, ref: EntrypointRef) -> "tuple[Any, ...]":
        raise NotImplementedError("resolve_toolset is reserved for a later phase")

    async def resolve_workflow(self, ref: EntrypointRef) -> Any:
        raise NotImplementedError("resolve_workflow is reserved for a later phase")


class PackageRegistry:
    """Default PackageSpecProvider AND PackageResourceProvider over a directory
    tree (spec §8.1): each subdirectory of ``base_dir`` is a package. The kind
    is read from an optional ``package.yaml``; otherwise it defaults to 'custom'.
    Resource list/read delegate to a DirectoryPackageResourceProvider so a single
    registry satisfies both Protocols."""

    def __init__(self, base_dir: "Path | str") -> None:
        self._base = Path(base_dir)
        # Lazily built; constructed on first resource access so PackageRegistry
        # stays cheap to instantiate when only spec listing is needed.
        self._resource_provider: "DirectoryPackageResourceProvider | None" = None

    def _resources(self) -> "DirectoryPackageResourceProvider":
        from .provider import DirectoryPackageResourceProvider
        if self._resource_provider is None:
            roots = {p.name: p for p in self._base.iterdir() if p.is_dir()} if self._base.is_dir() else {}
            self._resource_provider = DirectoryPackageResourceProvider(roots)
        return self._resource_provider

    async def list_ids(self) -> "tuple[str, ...]":
        if not self._base.is_dir():
            return ()
        return tuple(sorted(p.name for p in self._base.iterdir() if p.is_dir()))

    async def get(self, package_id: str) -> Any:
        from .spec import CapabilityPackageSpec

        root = self._base / package_id
        if not root.is_dir():
            raise PackageNotFoundError(f"package not found: {package_id}")
        kind = "custom"
        name = package_id
        pkg_yaml = root / "package.yaml"
        if pkg_yaml.is_file():
            from ..registry.parser import parse_yaml_text
            payload = parse_yaml_text(pkg_yaml.read_text(encoding="utf-8"), source=str(pkg_yaml))
            kind = str(payload.get("kind") or kind)
            name = str(payload.get("name") or name)
        return CapabilityPackageSpec(id=package_id, name=name, kind=kind)

    async def list_resources(self, scope, path: str = "", *, limit: int = 50,
                             cursor: "str | None" = None):
        return await self._resources().list_resources(scope, path, limit=limit, cursor=cursor)

    async def read_resource(self, ref, *, max_bytes: "int | None" = None):
        return await self._resources().read_resource(ref, max_bytes=max_bytes)


# Spec §8.4 alias.
DirectoryPackageRegistry = PackageRegistry
