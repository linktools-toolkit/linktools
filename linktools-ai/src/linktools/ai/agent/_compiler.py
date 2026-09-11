#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic Agent selection, binding, and exact historical recovery."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel

from ..capability import (
    CapabilityContribution,
    SkillDefinition,
    mcp_selector_server,
    mcp_server_namespace,
    mcp_server_selector,
)
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, ModelResolver
from ..spec import AgentSpec, AgentSpecCodec, MCPServerSpecCodec, SubagentRef
from ._binding import AgentBinding, AgentBindingSnapshot, SemanticPin
from ._definition import AgentDefinition
from ._output import bind_output, restore_output


class AgentCompiler:
    """Own the single Agent-level selection boundary for a frozen candidate set."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        candidates: Sequence[CapabilityContribution[object]],
        agents: Mapping[str, AgentSpec],
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        generic = tuple(item for item in candidates if item.kind == "capability")
        declarations = tuple(
            sorted(
                (item for item in candidates if item.kind != "capability"),
                key=lambda item: (item.kind, item.id),
            )
        )
        ordered = (*declarations, *generic)
        if len({(item.kind, item.id) for item in ordered}) != len(ordered):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        current_agents = dict(agents)
        if not current_agents:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        for agent_id, specification in current_agents.items():
            if not isinstance(agent_id, str) or not isinstance(specification, AgentSpec) or specification.id != agent_id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._models = model_resolver
        self._candidates = ordered
        self._by_identity = {(item.kind, item.id): item for item in ordered}
        self._agents: Mapping[str, AgentSpec] = MappingProxyType(current_agents)
        self._agent_ids = tuple(sorted(current_agents))
        self._mcp_by_namespace: dict[str, CapabilityContribution[object]] = {}
        for candidate in ordered:
            if candidate.kind != "mcp":
                continue
            namespace = mcp_server_namespace(candidate.id)
            if namespace in self._mcp_by_namespace:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            self._mcp_by_namespace[namespace] = candidate

    def compile(self, spec: AgentSpec) -> AgentDefinition:
        """Compile one current declaration from the frozen candidate universe."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        model = self._models.resolve(spec.model)
        selected_tools, selected_mcp, ordinary_policy, mcp_policy = self._select_tools(spec)
        selected_skills = self._select_exact_kind("skill", spec.allow_skills)
        selected_subagents = self._select_subagents(spec)
        selected_capabilities = self._select_exact_kind(
            "capability",
            spec.allow_capabilities,
        )
        return self._build_definition(
            spec,
            model=model,
            selected_tools=selected_tools,
            selected_skills=selected_skills,
            selected_mcp=selected_mcp,
            selected_capabilities=selected_capabilities,
            selected_subagents=selected_subagents,
            ordinary_policy=ordinary_policy,
            mcp_policy=mcp_policy,
        )

    def bind(
        self,
        definition: AgentDefinition,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        """Bind one root Definition to the exact durable output and child targets."""
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        subagents = tuple(
            SubagentRef(
                "agent",
                agent_id,
                self._agents[agent_id].description,
            )
            for agent_id in definition.selected_subagents
        )
        return self._bind(definition, output=output, subagents=subagents)

    def bind_subagent(
        self,
        definition: AgentDefinition,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        """Bind one child with delegation disabled for that execution."""
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        child_definition = self._build_definition(
            definition.spec,
            model=definition.model,
            selected_tools=definition.selected_tools,
            selected_skills=definition.selected_skills,
            selected_mcp=definition.selected_mcp,
            selected_capabilities=definition.selected_capabilities,
            selected_subagents=(),
            ordinary_policy=definition.ordinary_tool_policy,
            mcp_policy=definition.mcp_selector_policy,
        )
        return self._bind(child_definition, output=output, subagents=())

    def _bind(
        self,
        definition: AgentDefinition,
        *,
        output: "type[BaseModel] | None",
        subagents: Sequence[SubagentRef],
    ) -> AgentBinding:
        output_binding = bind_output(output)
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=AgentSpecCodec().from_payload(
                AgentSpecCodec().to_payload(definition.spec)
            ),
            base_model=dict(definition.model.semantic_payload),
            selected=tuple(_pin(candidate) for candidate in _semantic_candidates(definition)),
            subagents=tuple(subagents),
            output_mode=output_binding.mode,
            output_schema=output_binding.schema_definition,
        )
        return AgentBinding(
            snapshot.binding_digest,
            definition,
            output_binding,
            snapshot,
        )

    def restore(self, snapshot: AgentBindingSnapshot) -> AgentBinding:
        """Restore exact historical semantics without expanding current selectors."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        try:
            model = self._models.restore(
                snapshot.base_model,
                route_id=snapshot.agent_spec.model,
            )
            selected = self._restore_selected(snapshot.selected)
            ordinary_policy, mcp_policy = self._restore_policies(
                snapshot.agent_spec,
                selected["mcp"],
            )
            definition = self._build_definition(
                snapshot.agent_spec,
                model=model,
                selected_tools=selected["tool"],
                selected_skills=selected["skill"],
                selected_mcp=selected["mcp"],
                selected_capabilities=selected["capability"],
                selected_subagents=snapshot.subagent_ids,
                ordinary_policy=ordinary_policy,
                mcp_policy=mcp_policy,
            )
            output_binding = restore_output(snapshot.output_mode, snapshot.output_schema)
        except AIError as error:
            if error.code in {
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            }:
                raise
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        return AgentBinding(
            snapshot.binding_digest,
            definition,
            output_binding,
            snapshot,
        )

    def _restore_selected(
        self,
        pins: Sequence[SemanticPin],
    ) -> "dict[str, tuple[CapabilityContribution[object], ...]]":
        selected: dict[str, list[CapabilityContribution[object]]] = {
            "tool": [],
            "skill": [],
            "mcp": [],
            "capability": [],
        }
        for pin in pins:
            if pin.kind == "skill":
                value = SkillDefinition.from_semantic_contract(
                    cast("Mapping[str, object]", pin.contract)
                )
                candidate = CapabilityContribution("skill", pin.id, pin.fingerprint, value)
            elif pin.kind == "mcp":
                value = MCPServerSpecCodec().from_payload(cast("Mapping[str, object]", pin.contract))
                candidate = CapabilityContribution("mcp", pin.id, pin.fingerprint, value)
            else:
                current = self._by_identity.get((pin.kind, pin.id))
                if current is None or current.fingerprint != pin.fingerprint:
                    raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
                if current.semantic_contract != dict(pin.contract):
                    raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
                candidate = current
            selected[pin.kind].append(candidate)
        return {
            kind: (
                tuple(values)
                if kind == "capability"
                else tuple(sorted(values, key=lambda item: item.id))
            )
            for kind, values in selected.items()
        }

    def _select_tools(
        self,
        spec: AgentSpec,
    ) -> "tuple[tuple[CapabilityContribution[object], ...], tuple[CapabilityContribution[object], ...], tuple[str, ...], tuple[str, ...]]":
        tools = {
            candidate.id: candidate
            for candidate in self._candidates
            if candidate.kind == "tool"
        }
        if spec.allow_tools == ("*",):
            selected_tools = tuple(tools[name] for name in sorted(tools))
            selected_mcp = tuple(
                sorted(self._mcp_by_namespace.values(), key=lambda item: item.id)
            )
            return (
                selected_tools,
                selected_mcp,
                ("*",),
                tuple(f"{mcp_server_selector(item.id)}__*" for item in selected_mcp),
            )
        selected_tool_ids: set[str] = set()
        selected_mcp_by_id: dict[str, CapabilityContribution[object]] = {}
        ordinary_policy: list[str] = []
        mcp_policy: list[str] = []
        for selector in spec.allow_tools:
            parsed = mcp_selector_server(selector)
            if parsed is None:
                ordinary_policy.append(selector)
                if selector in tools:
                    selected_tool_ids.add(selector)
                continue
            namespace, _tool = parsed
            candidate = self._mcp_by_namespace.get(namespace)
            if candidate is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            selected_mcp_by_id[candidate.id] = candidate
            mcp_policy.append(selector)
        return (
            tuple(tools[name] for name in sorted(selected_tool_ids)),
            tuple(selected_mcp_by_id[name] for name in sorted(selected_mcp_by_id)),
            tuple(sorted(set(ordinary_policy))),
            tuple(sorted(set(mcp_policy))),
        )

    def _select_exact_kind(
        self,
        kind: Literal["skill", "capability"],
        selectors: Sequence[str],
    ) -> "tuple[CapabilityContribution[object], ...]":
        values = {
            candidate.id: candidate
            for candidate in self._candidates
            if candidate.kind == kind
        }
        if tuple(selectors) == ("*",):
            selected = tuple(
                candidate for candidate in self._candidates if candidate.kind == kind
            )
            return (
                selected
                if kind == "capability"
                else tuple(sorted(selected, key=lambda item: item.id))
            )
        selected_ids = set(selectors)
        if not selected_ids.issubset(values):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        selected = tuple(
            candidate
            for candidate in self._candidates
            if candidate.kind == kind and candidate.id in selected_ids
        )
        return (
            selected
            if kind == "capability"
            else tuple(sorted(selected, key=lambda item: item.id))
        )

    def _select_subagents(self, spec: AgentSpec) -> "tuple[str, ...]":
        available = set(self._agent_ids)
        if spec.allow_subagents == ("*",):
            return tuple(sorted(available.difference({spec.id})))
        selected = set(spec.allow_subagents)
        if spec.id in selected or not selected.issubset(available):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return tuple(sorted(selected))

    def _restore_policies(
        self,
        spec: AgentSpec,
        selected_mcp: Sequence[CapabilityContribution[object]],
    ) -> "tuple[tuple[str, ...], tuple[str, ...]]":
        if spec.allow_tools == ("*",):
            return (
                ("*",),
                tuple(
                    f"{mcp_server_selector(item.id)}__*"
                    for item in sorted(selected_mcp, key=lambda item: item.id)
                ),
            )
        ordinary = tuple(
            sorted(selector for selector in spec.allow_tools if not selector.startswith("mcp__"))
        )
        allowed_namespaces = {mcp_server_namespace(item.id) for item in selected_mcp}
        mcp_policy = []
        for selector in spec.allow_tools:
            parsed = mcp_selector_server(selector)
            if parsed is None:
                continue
            if parsed[0] not in allowed_namespaces:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            mcp_policy.append(selector)
        return ordinary, tuple(sorted(set(mcp_policy)))

    def _build_definition(
        self,
        spec: AgentSpec,
        *,
        model: ModelBinding,
        selected_tools: Sequence[CapabilityContribution[object]],
        selected_skills: Sequence[CapabilityContribution[object]],
        selected_mcp: Sequence[CapabilityContribution[object]],
        selected_capabilities: Sequence[CapabilityContribution[object]],
        selected_subagents: Sequence[str],
        ordinary_policy: Sequence[str],
        mcp_policy: Sequence[str],
    ) -> AgentDefinition:
        selected_skill_ids = {candidate.id for candidate in selected_skills}
        if any(skill_id not in selected_skill_ids for skill_id in spec.preload_skills):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        semantic = tuple(
            (
                *sorted(
                    (*selected_tools, *selected_skills, *selected_mcp),
                    key=lambda item: (item.kind, item.id),
                ),
                *selected_capabilities,
            )
        )
        digest = canonical_sha256(
            {
                "contract": "agent-definition-v1",
                "agent": AgentSpecCodec().to_payload(spec),
                "model_fingerprint": model.fingerprint,
                "selected": [
                    {"kind": item.kind, "id": item.id, "fingerprint": item.fingerprint}
                    for item in semantic
                ],
                "subagents": [
                    {"kind": "agent", "id": agent_id}
                    for agent_id in sorted(set(selected_subagents))
                ],
            }
        )
        return AgentDefinition(
            digest=digest,
            spec=spec,
            model=model,
            selected_tools=tuple(sorted(selected_tools, key=lambda item: item.id)),
            selected_skills=tuple(sorted(selected_skills, key=lambda item: item.id)),
            selected_mcp=tuple(sorted(selected_mcp, key=lambda item: item.id)),
            selected_capabilities=tuple(selected_capabilities),
            selected_subagents=tuple(sorted(set(selected_subagents))),
            ordinary_tool_policy=tuple(ordinary_policy),
            mcp_selector_policy=tuple(mcp_policy),
        )


def _semantic_candidates(
    definition: AgentDefinition,
) -> "tuple[CapabilityContribution[object], ...]":
    return tuple(
        (
            *sorted(
                (
                    *definition.selected_tools,
                    *definition.selected_skills,
                    *definition.selected_mcp,
                ),
                key=lambda item: (item.kind, item.id),
            ),
            *definition.selected_capabilities,
        )
    )


def _pin(candidate: CapabilityContribution[object]) -> SemanticPin:
    pin = SemanticPin(
        cast(Literal["tool", "skill", "mcp", "capability"], candidate.kind),
        candidate.id,
        candidate.semantic_contract,
    )
    if pin.fingerprint != candidate.fingerprint:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return pin


__all__ = ["AgentCompiler"]
