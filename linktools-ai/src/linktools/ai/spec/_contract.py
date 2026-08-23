"""Immutable declaration contracts for Agent and Asset specifications."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..core import ImmutableJsonMapping, JsonValue, canonical_string_tuple


@dataclass(frozen=True, slots=True)
class AgentUsageLimits:
    model_requests: "int | None" = None
    tool_calls: "int | None" = None
    input_tokens: "int | None" = None
    output_tokens: "int | None" = None
    total_tokens: "int | None" = None

    def __post_init__(self) -> None:
        values = (
            self.model_requests,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
        )
        if all(value is None for value in values):
            raise ValueError("usage limits must define at least one limit")
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
            for value in values
        ):
            raise ValueError("usage limits must be positive integers or None")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Durable, runtime-independent declaration of one Agent."""

    id: str
    revision: int
    model: str
    system_prompt: str = ""
    instructions: "tuple[str, ...]" = ()
    allow_tools: "tuple[str, ...]" = ("*",)
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)
    usage_limits: "AgentUsageLimits | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("agent id must be a non-empty string")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("agent revision must be a positive integer")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent model must be a non-empty string")
        if not isinstance(self.system_prompt, str):
            raise ValueError("agent system prompt must be a string")
        if isinstance(self.instructions, (str, bytes, bytearray)) or not isinstance(self.instructions, Sequence):
            raise ValueError("agent instructions must be a string array")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, str) for item in instructions):
            raise ValueError("agent instructions must be strings")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("agent metadata must be an object")
        if self.usage_limits is not None and not isinstance(self.usage_limits, AgentUsageLimits):
            raise ValueError("agent usage_limits must be AgentUsageLimits or None")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "allow_tools", canonical_string_tuple(self.allow_tools, field="allow_tools"))
        object.__setattr__(self, "metadata", ImmutableJsonMapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    revision: int
    content: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
            or not isinstance(self.content, str)
        ):
            raise ValueError("skill spec is incomplete")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    revision: int
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision < 1
            or not isinstance(self.command, str)
            or not self.command.strip()
            or isinstance(self.args, (str, bytes, bytearray))
            or not isinstance(self.args, Sequence)
        ):
            raise ValueError("MCP server spec is incomplete")
        args = tuple(self.args)
        if any(not isinstance(item, str) for item in args):
            raise ValueError("MCP server args must be strings")
        object.__setattr__(self, "args", args)


__all__ = ["AgentSpec", "AgentUsageLimits", "MCPServerSpec", "SkillSpec"]
