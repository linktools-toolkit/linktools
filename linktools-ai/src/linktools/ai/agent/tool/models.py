"""Core tool-domain values: declarations, invocation, policy spec, and operation state."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from ...governance.policy.rule import ApprovalMode, Permission, RiskLevel, SideEffectKind
from ...json import JsonValue, canonical_json_bytes, freeze_value, normalize_json
from ...storage.coordination.lease import Lease

if TYPE_CHECKING:
    from ..assembly.models import AgentFeatureRef
    from ..dependencies import AgentDependencies
    from ...execution.context import RunContext

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


class ToolTraceSink(Protocol):
    async def tool_result(self, payload: JsonValue) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    execution_id: str
    tool_call_id: str
    dependencies: "AgentDependencies"
    run_context: "RunContext | None" = None
    approved_tool_call_id: str | None = None
    approved_binding_fingerprint: str | None = None
    trace_sink: ToolTraceSink | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecuteTool:
    definition: ToolDefinition
    arguments: Mapping[str, JsonValue]
    context: ToolExecutionContext


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """An immutable tool declaration. Mirrors the policy layer's metadata
    shape; the codec fills it from YAML."""

    name: str
    description: str = ""
    enabled: bool = True
    permissions: "frozenset[Permission]" = field(
        default_factory=lambda: frozenset({Permission.READ})
    )
    risk: RiskLevel = RiskLevel.LOW
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY
    approval: ApprovalMode = ApprovalMode.NEVER
    idempotent: "bool | None" = None
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    idempotency_strategy: "str | None" = None
    idempotency_key_field: "str | None" = None
    # bump when a tool's input contract changes shape so an idempotency
    # hash computed under the old schema is never mistaken for a match
    # against the new one (see idempotency.compute_request_hash).
    schema_version: str = "1"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


class ToolOperationStatus(str, Enum):
    PREPARED = "prepared"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class ToolOperation:
    id: str
    tenant_id: str | None
    execution_id: str
    tool_call_id: str
    idempotency_key: str
    tool_name: str
    arguments_hash: str
    binding_fingerprint: str
    status: ToolOperationStatus
    replay_safe: bool = False
    lease: Lease = Lease()
    result: JsonValue | None = None
    error: JsonValue | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def owner(self) -> str | None:
        return self.lease.owner

    @property
    def fence(self) -> int:
        return self.lease.fence

    @property
    def lease_expires_at(self) -> datetime | None:
        return self.lease.expires_at
