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
            value is not None and (not isinstance(value, int) or isinstance(value, bool))
            for value in values
        ):
            raise TypeError("usage limits must contain integers or None")
        if any(value is not None and value <= 0 for value in values):
            raise ValueError("usage limits must be positive integers")


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
        if not isinstance(self.id, str):
            raise TypeError("agent id must be a string")
        if not self.id.strip():
            raise ValueError("agent id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("agent revision must be an integer")
        if self.revision < 1:
            raise ValueError("agent revision must be positive")
        if not isinstance(self.model, str):
            raise TypeError("agent model must be a string")
        if not self.model.strip():
            raise ValueError("agent model must be non-empty")
        if not isinstance(self.system_prompt, str):
            raise TypeError("agent system prompt must be a string")
        if isinstance(self.instructions, (str, bytes, bytearray)) or not isinstance(self.instructions, Sequence):
            raise TypeError("agent instructions must be a string array")
        instructions = tuple(self.instructions)
        if any(not isinstance(item, str) for item in instructions):
            raise TypeError("agent instructions must be strings")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("agent metadata must be an object")
        if self.usage_limits is not None and not isinstance(self.usage_limits, AgentUsageLimits):
            raise TypeError("agent usage_limits must be AgentUsageLimits or None")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "allow_tools", canonical_string_tuple(self.allow_tools, field="allow_tools"))
        object.__setattr__(self, "metadata", ImmutableJsonMapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    revision: int
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("skill id must be a string")
        if not self.id.strip():
            raise ValueError("skill id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("skill revision must be an integer")
        if self.revision < 1:
            raise ValueError("skill revision must be positive")
        if not isinstance(self.content, str):
            raise TypeError("skill content must be a string")


@dataclass(frozen=True, slots=True)
class MCPServerSpec:
    id: str
    revision: int
    command: str
    args: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("MCP server id must be a string")
        if not self.id.strip():
            raise ValueError("MCP server id must be non-empty")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("MCP server revision must be an integer")
        if self.revision < 1:
            raise ValueError("MCP server revision must be positive")
        if not isinstance(self.command, str):
            raise TypeError("MCP server command must be a string")
        if not self.command.strip():
            raise ValueError("MCP server command must be non-empty")
        if isinstance(self.args, (str, bytes, bytearray)) or not isinstance(self.args, Sequence):
            raise TypeError("MCP server args must be a string sequence")
        args = tuple(self.args)
        if any(not isinstance(item, str) for item in args):
            raise TypeError("MCP server args must be strings")
        object.__setattr__(self, "args", args)


__all__ = ["AgentSpec", "AgentUsageLimits", "MCPServerSpec", "SkillSpec"]
