"""Core model-callable tool values."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Mapping

from ...governance.policy.rule import RiskLevel, SideEffectKind
from ...json import JsonValue, canonical_json_bytes, freeze_value, normalize_json

if TYPE_CHECKING:
    from ..assembly.models import AgentFeatureRef

ToolHandler = Callable[..., Awaitable[object]]


class ToolSource(StrEnum):
    BUILTIN = "builtin"
    MCP = "mcp"
    SKILL = "skill"
    SUBAGENT = "subagent"
    EXTENSION = "extension"


class ToolCategory(StrEnum):
    DISCOVERY = "discovery"
    FILE_READ = "file-read"
    FILE_WRITE = "file-write"
    NETWORK_READ = "network-read"
    NETWORK_WRITE = "network-write"
    TERMINAL = "terminal"
    SUBAGENT = "subagent"
    EXTENSION_READ = "extension-read"
    EXTENSION_EXECUTE = "extension-execute"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    source: ToolSource
    category: ToolCategory
    risk: RiskLevel
    side_effect: SideEffectKind
    feature: "AgentFeatureRef"
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolDescriptor.name must not be empty")
        object.__setattr__(
            self,
            "metadata",
            freeze_value(normalize_json(dict(self.metadata))),
        )

    @property
    def mutating(self) -> bool:
        return self.side_effect not in {
            SideEffectKind.NONE,
            SideEffectKind.READ_ONLY,
        }

    def fingerprint(self) -> str:
        payload: JsonValue = {
            "name": self.name,
            "source": self.source.value,
            "category": self.category.value,
            "risk": self.risk.value,
            "side_effect": self.side_effect.value,
            "feature": {
                "kind": self.feature.kind,
                "name": self.feature.name,
                "config": dict(self.feature.config),
            },
            "metadata": dict(self.metadata),
        }
        return sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    descriptor: ToolDescriptor
    handler: ToolHandler
    input_schema: Mapping[str, JsonValue] | None = None
    description: str | None = None
    input_schema_version: str = "1"
    provider_revision: str = ""
    handler_revision: str = ""


@dataclass(slots=True)
class ToolHandlerSet:
    """Provider-owned handlers before the sole SDK adapter boundary."""

    handlers: dict[str, ToolHandler] = field(default_factory=dict)

    def add_function(self, handler: ToolHandler) -> None:
        self.handlers[handler.__name__] = handler


def declared_tool_definitions(
    handler_set: ToolHandlerSet,
    descriptors: tuple[ToolDescriptor, ...],
) -> tuple[ToolDefinition, ...]:
    """Pair provider-native handlers with their internal declarations."""
    handlers = handler_set.handlers
    declared = {descriptor.name for descriptor in descriptors}
    actual = set(handlers)
    if declared != actual:
        raise ValueError(
            f"tool descriptor mismatch: missing={sorted(declared - actual)}, "
            f"extra={sorted(actual - declared)}"
        )
    return tuple(
        ToolDefinition(
            descriptor=descriptor,
            handler=handlers[descriptor.name],
            description=handlers[descriptor.name].__doc__,
        )
        for descriptor in descriptors
    )
