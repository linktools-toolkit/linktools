#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpec / MCPRegistry: MCP server definitions loaded from mcp.yaml."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Self

from ..core.registry import BaseRegistry, BaseSpec, SpecSource, find_file
from ..support.config import (
    load_yaml_file as _load_yaml_file,
    load_yaml_text as _load_yaml_text,
)

if TYPE_CHECKING:
    from ..resource_store.store import ResourceStore

logger = logging.getLogger("linktools.ai.mcp.registry")


@dataclass(slots=True)
class MCPServerSpec(BaseSpec):
    description: str = ""
    server_name: str = ""
    kind: str = "read"
    provides: "list[str]" = field(default_factory=list)
    mcp_type: str = "stdio"
    command: str = ""
    args: "list[str]" = field(default_factory=list)
    env: "dict[str, str]" = field(default_factory=dict)
    url: str = ""
    headers: "dict[str, str]" = field(default_factory=dict)
    circuit_breaker: "dict[str, object]" = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: "Mapping[str, object]", source: SpecSource) -> Self:
        mcp = dict(payload.get("mcp", {}))
        return cls(
            name=source.name,
            path=source.path,
            base_dir=source.base_dir,
            enabled=bool(payload.get("enabled", mcp.get("enabled", True))),
            description=str(payload.get("description") or payload.get("display_name") or ""),
            server_name=str(mcp.get("server", source.name)),
            kind=str(payload.get("kind", "read")),
            provides=list(payload.get("provides", [])),
            mcp_type=str(mcp.get("type", "stdio")),
            command=str(mcp.get("command", "")),
            args=[str(a) for a in mcp.get("args", [])],
            env={str(k): str(v) for k, v in mcp.get("env", {}).items()},
            url=str(mcp.get("url", "")),
            headers={str(k): str(v) for k, v in mcp.get("headers", {}).items()},
            circuit_breaker=dict(payload.get("circuit_breaker", {})),
        )


class MCPRegistry(BaseRegistry[MCPServerSpec]):
    """Scan MCP directories and load MCPServerSpec objects keyed by server_id."""

    def __init__(self, *paths: Path, resource_store: "ResourceStore | None" = None) -> None:
        super().__init__(*paths)
        self._resource_store = resource_store

    async def _load(self) -> "dict[str, MCPServerSpec]":
        result: "dict[str, MCPServerSpec]" = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                spec = None
                if child.is_file() and child.name.lower() == "mcp.yaml":
                    spec = self._load_spec(child.stem, child, None)
                elif child.is_dir():
                    yaml_file = find_file(child, "mcp.yaml")
                    if yaml_file is not None:
                        spec = self._load_spec(child.name, yaml_file, child)
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("MCP server '%s' from %s overrides existing registration", spec.name, child)
                    result[spec.name] = spec
                    logger.debug("mcp loaded: name=%s path=%s", spec.name, child)
        if self._resource_store is not None:
            for resource in await self._resource_store.get_by_name("mcp", "mcp.yaml"):
                capability_id = resource.path.split("/")[2]
                content = resource.content
                payload = _load_yaml_text(content, source=f"<db:{capability_id}>")
                base_dir = next(
                    (p / capability_id for p in self._paths if (p / capability_id).is_dir()), None
                )
                runtime_name = self._runtime_name_from_capability(capability_id, payload.get("name"), label="MCP server")
                spec = MCPServerSpec.from_dict(
                    payload,
                    SpecSource(
                        name=runtime_name,
                        path=(base_dir / "mcp.yaml") if base_dir else Path(f"adapter/{capability_id}/mcp.yaml"),
                        base_dir=base_dir,
                    ),
                )
                if spec.enabled:
                    if spec.name in result:
                        logger.warning("MCP server '%s' from DB overrides source registration", spec.name)
                    result[spec.name] = spec
                    logger.debug("mcp loaded from DB: name=%s", spec.name)
        logger.debug("MCPRegistry: loaded %d servers [%s]", len(result), ", ".join(result))
        return result

    def _load_spec(self, name: str, path: Path, base_dir: "Path | None") -> MCPServerSpec:
        payload = _load_yaml_file(path)
        return MCPServerSpec.from_dict(
            payload,
            SpecSource(
                name=str(payload.get("name") or name),
                path=path,
                base_dir=base_dir,
            )
        )

    def resolve_by_capability(self, capability: str) -> "MCPServerSpec | None":
        """Find the first registered server whose `provides` list contains `capability`."""
        for spec in self:
            if capability in spec.provides:
                return spec
        return None
