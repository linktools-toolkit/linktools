"""Assembly-time tool exposure and schema enforcement."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ...errors import ToolConflictError
from ..assembly.models import AgentFeatureRef
from .schema import ToolSchemaValidator
from .models import ToolCategory, ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolExposurePolicy:
    expose_discovery_tools: bool = True
    expose_execution_tools: bool = False
    max_tools_total: int = 64
    max_tools_per_feature: int = 16


class ToolAssembler:
    def __init__(
        self,
        *,
        exposure: ToolExposurePolicy,
        schema_validator: ToolSchemaValidator,
    ) -> None:
        self._exposure = exposure
        self._schema_validator = schema_validator

    def assemble(
        self,
        definitions: Iterable[ToolDefinition],
        *,
        owner_by_definition: Mapping[int, AgentFeatureRef],
    ) -> tuple[ToolDefinition, ...]:
        exposed: list[ToolDefinition] = []
        owner_counts: Counter[tuple[str, str]] = Counter()
        names: set[str] = set()
        for definition in definitions:
            if definition.input_schema is not None:
                self._schema_validator.validate_schema(definition.input_schema)
            descriptor = definition.descriptor
            if (
                descriptor.category is ToolCategory.DISCOVERY
                and not self._exposure.expose_discovery_tools
            ):
                continue
            if descriptor.mutating and not self._exposure.expose_execution_tools:
                continue
            if descriptor.name in names:
                raise ToolConflictError(
                    f"duplicate exposed tool name {descriptor.name!r}"
                )
            owner = owner_by_definition[id(definition)]
            owner_key = (owner.kind, owner.name)
            owner_counts[owner_key] += 1
            if owner_counts[owner_key] > self._exposure.max_tools_per_feature:
                raise ToolConflictError(
                    f"feature {owner} exceeds max_tools_per_feature="
                    f"{self._exposure.max_tools_per_feature}"
                )
            names.add(descriptor.name)
            exposed.append(definition)
        if len(exposed) > self._exposure.max_tools_total:
            raise ToolConflictError(
                f"tool count exceeds max_tools_total={self._exposure.max_tools_total}"
            )
        return tuple(exposed)
