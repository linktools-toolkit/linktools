"""Immutable executable Agent definitions."""

from dataclasses import dataclass

from pydantic import BaseModel

from ..capability import CapabilityBinding, validate_fingerprint
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec
from ._binding import AgentBindingSnapshot
from ._output import OutputBinding

_TRUSTED_TOOL_CLASSES = frozenset(
    {
        "control",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "memory.read",
        "memory.write",
    }
)


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Freeze all declaration, model, output, and capability execution inputs."""

    digest: str
    spec: AgentSpec
    model: ModelBinding
    output_binding: OutputBinding
    effective_capabilities: "tuple[CapabilityBinding, ...]"
    binding_snapshot: AgentBindingSnapshot
    trusted_tool_classes: "tuple[tuple[str, str], ...]" = ()
    trusted_mcp_selectors: "tuple[str, ...]" = ()

    def __post_init__(self) -> None:
        validate_fingerprint(self.digest)
        if not isinstance(self.output_binding, OutputBinding):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(self.binding_snapshot, AgentBindingSnapshot):
            raise AIError(ErrorCode.BINDING_CONFLICT)
        if (
            self.binding_snapshot.binding_digest != self.digest
            or self.binding_snapshot.agent_spec != self.spec
            or self.binding_snapshot.output_schema_id != self.output_binding.schema_id
            or self.binding_snapshot.output_schema_revision != self.output_binding.schema_revision
            or self.binding_snapshot.output_schema_fingerprint != self.output_binding.schema_fingerprint
            or self.binding_snapshot.output_type_module != self.output_type.__module__
            or self.binding_snapshot.output_type_qualname != self.output_type.__qualname__
        ):
            raise AIError(ErrorCode.BINDING_CONFLICT)
        if any(capability is None for capability in self.effective_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            identities = tuple((capability.provider, capability.id) for capability in self.effective_capabilities)
        except (AttributeError, TypeError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        selectors: set[str] = set()
        previous_selector: str | None = None
        for selector in self.trusted_mcp_selectors:
            if (
                not isinstance(selector, str)
                or not selector.startswith("mcp__")
                or selector == "mcp__"
                or "__" in selector[5:]
                or selector in selectors
                or (previous_selector is not None and selector < previous_selector)
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            selectors.add(selector)
            previous_selector = selector
        names: set[str] = set()
        previous: str | None = None
        for name, tool_class in self.trusted_tool_classes:
            if (
                not isinstance(name, str)
                or not name.strip()
                or tool_class not in _TRUSTED_TOOL_CLASSES
                or name in names
                or (previous is not None and name < previous)
            ):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            names.add(name)
            previous = name

    @property
    def output_type(self) -> type[BaseModel]:
        return self.output_binding.value_type

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_binding.schema_fingerprint


__all__ = ["AgentDefinition"]
