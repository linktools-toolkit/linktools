"""Immutable values shared by feature providers and the assembler."""

from dataclasses import dataclass, field
from collections.abc import Sequence
from hashlib import sha256
from typing import Mapping

from ...json import JsonValue, freeze_value, normalize_json
from ...json import canonical_json_bytes
from ..tool.models import ToolDefinition


@dataclass(frozen=True, slots=True)
class AgentFeatureRef:
    """A declaration-level reference to an agent feature."""

    kind: str
    name: str
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("AgentFeatureRef.kind must not be empty")
        if not self.name.strip():
            raise ValueError("AgentFeatureRef.name must not be empty")
        object.__setattr__(
            self,
            "config",
            freeze_value(normalize_json(dict(self.config))),
        )

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"

    def fingerprint(self) -> str:
        return sha256(
            canonical_json_bytes(
                {
                    "kind": self.kind,
                    "name": self.name,
                    "config": dict(self.config),
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentContribution:
    """The prompt and tools contributed by one feature provider."""

    prompt_sections: Mapping[str, str] = field(default_factory=dict)
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_sections",
            freeze_value(dict(self.prompt_sections)),
        )
        if not isinstance(self.tools, tuple):
            raise TypeError("AgentContribution.tools must be a tuple")

    @classmethod
    def empty(cls) -> "AgentContribution":
        return cls()


@dataclass(frozen=True, slots=True)
class AgentAssembly:
    """The final prompt and tool surface for one execution."""

    prompt_sections: Mapping[str, str]
    tools: tuple[ToolDefinition, ...]
    feature_owners: Mapping[str, AgentFeatureRef]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_sections",
            freeze_value(dict(self.prompt_sections)),
        )
        object.__setattr__(
            self,
            "feature_owners",
            freeze_value(dict(self.feature_owners)),
        )


def parse_agent_feature_refs(
    items: object,
) -> tuple[AgentFeatureRef, ...]:
    """Parse the shared feature declaration shape."""
    if items is None:
        return ()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("features must be a sequence")
    refs: list[AgentFeatureRef] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("feature declarations must be mappings")
        unknown = set(item) - {"kind", "name", "config"}
        if unknown:
            raise ValueError(f"unknown feature fields: {sorted(unknown)}")
        refs.append(
            AgentFeatureRef(
                kind=str(item.get("kind", "")),
                name=str(item.get("name", "")),
                config=item.get("config") or {},
            )
        )
    return tuple(refs)
