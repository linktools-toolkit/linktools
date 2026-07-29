"""Handler-free inspection values for assembled agents."""

from dataclasses import dataclass

from .models import AgentAssembly, AgentFeatureRef
from ..tool.models import ToolDescriptor


@dataclass(frozen=True, slots=True)
class AgentInspection:
    features: tuple[AgentFeatureRef, ...]
    tools: tuple[ToolDescriptor, ...]
    prompt_section_names: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_assembly(
        cls,
        assembly: AgentAssembly,
        *,
        features: tuple[AgentFeatureRef, ...],
        warnings: tuple[str, ...] = (),
    ) -> "AgentInspection":
        return cls(
            features=features,
            tools=tuple(definition.descriptor for definition in assembly.tools),
            prompt_section_names=tuple(assembly.prompt_sections),
            warnings=warnings,
        )
