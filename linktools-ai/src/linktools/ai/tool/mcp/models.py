"""Dependency-free MCP connection and discovery value types."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...errors import MCPToolDefinitionError
from ...json import freeze_value
from ..schema import validate_schema


@dataclass(frozen=True, slots=True)
class MCPConnectionRef:
    server_id: str
    fingerprint: str


@dataclass(frozen=True)
class MCPExposedTool:
    server_id: str
    raw_name: str
    exposed_name: str
    parameters_json_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    read_only: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters_json_schema",
            freeze_value(dict(self.parameters_json_schema)),
        )
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    parameters_json_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    read_only: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise MCPToolDefinitionError("MCP tool name must be non-empty")
        try:
            validate_schema(self.parameters_json_schema)
        except Exception as exc:
            raise MCPToolDefinitionError(
                f"invalid schema for MCP tool {self.name!r}"
            ) from exc
        object.__setattr__(
            self,
            "parameters_json_schema",
            freeze_value(dict(self.parameters_json_schema)),
        )
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


@dataclass(frozen=True)
class MCPDiscoveryResult:
    tools: tuple[MCPToolInfo, ...] = ()
    verified: bool = False
    error: BaseException | None = None
    connection_ref: MCPConnectionRef | None = None


__all__ = [
    "MCPConnectionRef",
    "MCPDiscoveryResult",
    "MCPExposedTool",
    "MCPToolInfo",
]
