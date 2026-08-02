#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The single boundary that turns feature declarations into an agent surface."""

from typing import TYPE_CHECKING
from linktools.core import environ
from ...errors import AgentAssemblyError, AgentFeatureConflictError, AgentFeatureNotFoundError
from .models import AgentAssembly, AgentFeatureRef
from .provider import AgentAssemblyEventSink

logger = environ.get_logger("ai.agent.assembly.assembler")

if TYPE_CHECKING:
    from ..tool.exposure import ToolAssembler
    from .provider import AgentFeatureContext
    from .registry import AgentFeatureRegistry
    from ..spec import AgentSpec


class AgentAssembler:
    def __init__(
        self,
        *,
        registry: "AgentFeatureRegistry",
        tool_assembler: "ToolAssembler",
        events: "AgentAssemblyEventSink | None" = None,
    ) -> None:
        self._registry = registry
        self._tool_assembler = tool_assembler
        self._events = events

    def validate_features(self, spec: "AgentSpec") -> None:
        seen: "set[tuple[str, str]]" = set()
        for ref in spec.features:
            key = (ref.kind, ref.name)
            if key in seen:
                raise AgentFeatureConflictError(
                    f"agent {spec.id}: duplicate feature declaration {ref}"
                )
            seen.add(key)
            if self._registry.get(ref.kind) is None:
                raise AgentFeatureNotFoundError(
                    f"agent {spec.id}: no provider registered for {ref}"
                )

    async def assemble(
        self,
        spec: "AgentSpec",
        context: "AgentFeatureContext",
    ) -> AgentAssembly:
        self.validate_features(spec)
        seen: "set[tuple[str, str]]" = set()
        prompt_sections: "dict[str, str]" = {}
        definitions = []
        owner_by_definition: "dict[int, AgentFeatureRef]" = {}

        async def emit(event: object) -> None:
            sinks = tuple(
                sink
                for sink in (self._events, context.events)
                if sink is not None
            )
            seen_sinks: "set[int]" = set()
            for sink in sinks:
                if id(sink) in seen_sinks:
                    continue
                seen_sinks.add(id(sink))
                await sink.emit(event)

        for ref in spec.features:
            key = (ref.kind, ref.name)
            if key in seen:
                raise AgentFeatureConflictError(
                    f"agent {spec.id}: duplicate feature declaration {ref}"
                )
            seen.add(key)
            provider = self._registry.get(ref.kind)
            if provider is None:
                raise AgentFeatureNotFoundError(
                    f"agent {spec.id}: no provider registered for {ref}"
                )
            from ...observability.events.payloads import (
                AgentFeatureResolveCompleted,
                AgentFeatureResolveStarted,
            )

            await emit(
                AgentFeatureResolveStarted(
                    agent_id=spec.id,
                    feature_ref=f"{ref.kind}:{ref.name}",
                )
            )
            contribution = await provider.resolve(ref, context)
            if environ.debug:
                logger.debug(
                    "feature %s:%s resolved: tools=%s prompt_sections=%s",
                    ref.kind, ref.name, len(contribution.tools), tuple(contribution.prompt_sections),
                )
            await emit(
                AgentFeatureResolveCompleted(
                    agent_id=spec.id,
                    feature_ref=f"{ref.kind}:{ref.name}",
                    tool_count=len(contribution.tools),
                )
            )
            for name, text in contribution.prompt_sections.items():
                if not text:
                    continue
                existing = prompt_sections.get(name)
                if existing is None:
                    prompt_sections[name] = text
                elif existing != text:
                    prompt_sections[name] = f"{existing}\n{text}"
            for definition in contribution.tools:
                if definition.descriptor.feature != ref:
                    raise AgentAssemblyError(
                        f"agent {spec.id}: provider for {ref} contributed "
                        f"tool {definition.descriptor.name!r} owned by "
                        f"{definition.descriptor.feature}"
                    )
                owner_by_definition[id(definition)] = ref
                definitions.append(definition)

        tools = self._tool_assembler.assemble(
            definitions,
            owner_by_definition=owner_by_definition,
        )
        exposed_owners = {
            definition.descriptor.name: owner_by_definition[id(definition)]
            for definition in tools
        }
        return AgentAssembly(
            prompt_sections=prompt_sections,
            tools=tools,
            feature_owners=exposed_owners,
        )
