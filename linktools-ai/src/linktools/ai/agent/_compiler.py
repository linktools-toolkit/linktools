#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic Agent selection, binding, and exact historical recovery."""

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel

from ..capability import (
    CapabilityContribution,
    capability_fingerprint,
    mcp_selector_server,
    mcp_server_namespace,
    mcp_server_selector,
)
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, ModelResolver
from ..spec import AgentSpec, AgentSpecCodec, MCPServerSpecCodec, SkillSpecCodec
from ._binding import AgentBinding, AgentBindingSnapshot, SemanticPin, SubagentRef
from ._definition import AgentDefinition
from ._output import bind_output, restore_output


class AgentCompiler:
    """Own the single Agent-level selection boundary for a frozen candidate set."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        candidates: Sequence[CapabilityContribution[object]],
        agent_ids: Sequence[str],
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        ordered = tuple(sorted(candidates, key=lambda item: (item.kind, item.id)))
        if len({(item.kind, item.id) for item in ordered}) != len(ordered):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._models = model_resolver
        self._candidates = ordered
        self._by_identity = {(item.kind, item.id): item for item in ordered}
        self._agent_ids = tuple(sorted(set(agent_ids)))
        if not self._agent_ids:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
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
        selected_capabilities = tuple(
            candidate for candidate in self._candidates if candidate.kind == "capability"
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
        """Bind one current Definition to the exact durable output contract."""
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        output_binding = bind_output(output)
        digest = _binding_digest(definition.digest, output_binding.fingerprint)
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=definition.spec,
            model=dict(definition.model.semantic_payload),
            selected=tuple(
                sorted(
                    (_pin(candidate) for candidate in _semantic_candidates(definition)),
                    key=lambda item: (item.kind, item.id),
                )
            ),
            subagents=tuple(SubagentRef("agent", agent_id) for agent_id in definition.selected_subagents),
            output_mode=output_binding.mode,
            output_schema=output_binding.schema_definition,
            binding_digest=digest,
        )
        return AgentBinding(digest, definition, output_binding, snapshot)

    def restore(self, snapshot: AgentBindingSnapshot) -> AgentBinding:
        """Restore exact historical semantics without expanding current selectors."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        try:
            model = self._models.restore(snapshot.model, route_id=snapshot.agent_spec.model)
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
            if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                raise
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        digest = _binding_digest(definition.digest, output_binding.fingerprint)
        if digest != snapshot.binding_digest:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return AgentBinding(digest, definition, output_binding, snapshot)

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
                value = SkillSpecCodec().from_payload(cast("Mapping[str, object]", pin.contract))
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
            kind: tuple(sorted(values, key=lambda item: item.id))
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
        kind: Literal["skill"],
        selectors: Sequence[str],
    ) -> "tuple[CapabilityContribution[object], ...]":
        values = {
            candidate.id: candidate
            for candidate in self._candidates
            if candidate.kind == kind
        }
        if tuple(selectors) == ("*",):
            return tuple(values[name] for name in sorted(values))
        selected: list[CapabilityContribution[object]] = []
        for selector in selectors:
            candidate = values.get(selector)
            if candidate is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            selected.append(candidate)
        return tuple(sorted(selected, key=lambda item: item.id))

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
        semantic = tuple(
            sorted(
                (*selected_tools, *selected_skills, *selected_mcp, *selected_capabilities),
                key=lambda item: (item.kind, item.id),
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
            selected_capabilities=tuple(sorted(selected_capabilities, key=lambda item: item.id)),
            selected_subagents=tuple(sorted(set(selected_subagents))),
            ordinary_tool_policy=tuple(ordinary_policy),
            mcp_selector_policy=tuple(mcp_policy),
        )


def _semantic_candidates(
    definition: AgentDefinition,
) -> "tuple[CapabilityContribution[object], ...]":
    return tuple(
        sorted(
            (
                *definition.selected_tools,
                *definition.selected_skills,
                *definition.selected_mcp,
                *definition.selected_capabilities,
            ),
            key=lambda item: (item.kind, item.id),
        )
    )


def _pin(candidate: CapabilityContribution[object]) -> SemanticPin:
    contract = candidate.semantic_contract
    if capability_fingerprint(candidate.kind, candidate.id, contract) != candidate.fingerprint:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return SemanticPin(
        cast(Literal["tool", "skill", "mcp", "capability"], candidate.kind),
        candidate.id,
        1,
        contract,
    )


def _binding_digest(agent_definition_digest: str, output_fingerprint: str) -> str:
    return canonical_sha256(
        {
            "contract": "agent-binding-v1",
            "agent_definition_digest": agent_definition_digest,
            "output_fingerprint": output_fingerprint,
        }
    )


__all__ = ["AgentCompiler"]
