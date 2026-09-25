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
    workspace_tool_declarations,
)
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, ModelResolver
from ..spec import (
    AgentSpec,
    AgentSpecCodec,
    SubagentRef,
    mcp_server_selector,
    parse_mcp_tool_selector,
)
from ._binding import AgentBinding, AgentBindingContract, CapabilityPin
from ._compiled import CompiledAgent
from ._output import bind_output, restore_output


class AgentCompiler:
    """Own the single Agent-level selection boundary for a captured candidate set."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        candidates: Sequence[CapabilityContribution[object]],
        agents: Mapping[str, AgentSpec],
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        generic = tuple(item for item in candidates if item.kind == "runtime_capability")
        declarations = tuple(
            sorted(
                (item for item in candidates if item.kind != "runtime_capability"),
                key=lambda item: (item.kind, item.id),
            )
        )
        ordered = (*declarations, *generic)
        if len({(item.kind, item.id) for item in ordered}) != len(ordered):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        current_agents = dict(agents)
        for agent_id, specification in current_agents.items():
            if not isinstance(agent_id, str) or not isinstance(specification, AgentSpec) or specification.id != agent_id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._models = model_resolver
        self._candidates = ordered
        self._by_identity = {(item.kind, item.id): item for item in ordered}
        self._agents: Mapping[str, AgentSpec] = MappingProxyType(current_agents)
        self._agent_ids = tuple(sorted(current_agents))
        self._mcp_by_id: dict[str, CapabilityContribution[object]] = {}
        server_tokens: dict[str, str] = {}
        for candidate in ordered:
            if candidate.kind == "tool" and candidate.id.startswith("mcp__"):
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            if candidate.kind != "mcp":
                continue
            token = canonical_sha256(
                {
                    "version": 1,
                    "kind": "mcp-server-name",
                    "server_id": candidate.id,
                }
            )[:24]
            previous = server_tokens.get(token)
            if previous is not None and previous != candidate.id:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            server_tokens[token] = candidate.id
            self._mcp_by_id[candidate.id] = candidate

    def compile(self, spec: AgentSpec) -> CompiledAgent:
        """Compile one current declaration from the captured candidate universe."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        model = self._models.resolve(spec.model_route)
        selected_tools, selected_mcp, ordinary_policy, mcp_policy = self._select_tools(spec)
        selected_skills = self._select_exact_kind("skill", spec.allow_skills)
        selected_subagents = self._select_subagents(spec)
        selected_runtime_capabilities = self._select_exact_kind(
            "runtime_capability",
            spec.allow_runtime_capabilities,
        )
        return self._build_compiled_agent(
            spec,
            model=model,
            selected_tools=selected_tools,
            selected_skills=selected_skills,
            selected_mcp=selected_mcp,
            selected_runtime_capabilities=selected_runtime_capabilities,
            selected_subagents=selected_subagents,
            ordinary_policy=ordinary_policy,
            mcp_policy=mcp_policy,
        )

    def bind(
        self,
        compiled_agent: CompiledAgent,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        """Bind one CompiledAgent to durable output and child targets."""
        if not isinstance(compiled_agent, CompiledAgent):
            raise TypeError("compiled_agent must be CompiledAgent")
        subagents = tuple(
            SubagentRef(
                "agent",
                agent_id,
                self._agents[agent_id].description,
                revision=self._agents[agent_id].revision,
            )
            for agent_id in compiled_agent.selected_subagents
        )
        return self._bind(compiled_agent, output=output, subagents=subagents)

    def bind_subagent(
        self,
        compiled_agent: CompiledAgent,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        """Bind one child with delegation disabled for that execution."""
        if not isinstance(compiled_agent, CompiledAgent):
            raise TypeError("compiled_agent must be CompiledAgent")
        child_compiled_agent = self._build_compiled_agent(
            compiled_agent.spec,
            model=compiled_agent.model,
            selected_tools=compiled_agent.selected_tools,
            selected_skills=compiled_agent.selected_skills,
            selected_mcp=compiled_agent.selected_mcp,
            selected_runtime_capabilities=compiled_agent.selected_runtime_capabilities,
            selected_subagents=(),
            ordinary_policy=compiled_agent.ordinary_tool_policy,
            mcp_policy=compiled_agent.mcp_selector_policy,
        )
        return self._bind(child_compiled_agent, output=output, subagents=())

    def _bind(
        self,
        compiled_agent: CompiledAgent,
        *,
        output: "type[BaseModel] | None",
        subagents: Sequence[SubagentRef],
    ) -> AgentBinding:
        output_binding = bind_output(output)
        binding_contract = AgentBindingContract(
            agent_spec=AgentSpecCodec().from_payload(
                AgentSpecCodec().to_payload(compiled_agent.spec)
            ),
            model_contract=dict(compiled_agent.model.contract),
            selected=tuple(_pin(candidate) for candidate in _selected_candidates(compiled_agent)),
            subagents=tuple(subagents),
            output_mode=output_binding.mode,
            output_schema=output_binding.schema_definition,
        )
        return AgentBinding(
            compiled_agent,
            output_binding,
            binding_contract,
        )

    def restore(self, binding_contract: AgentBindingContract) -> AgentBinding:
        """Restore exact historical semantics without expanding current selectors."""
        if not isinstance(binding_contract, AgentBindingContract):
            raise TypeError("binding_contract must be AgentBindingContract")
        try:
            model = self._models.restore(
                binding_contract.model_contract,
                route_id=binding_contract.agent_spec.model_route,
            )
            selected = self._restore_selected(binding_contract.selected)
            ordinary_policy, mcp_policy = self._restore_policies(
                binding_contract.agent_spec,
                selected["tool"],
                selected["mcp"],
            )
            compiled_agent = self._build_compiled_agent(
                binding_contract.agent_spec,
                model=model,
                selected_tools=selected["tool"],
                selected_skills=selected["skill"],
                selected_mcp=selected["mcp"],
                selected_runtime_capabilities=selected["runtime_capability"],
                selected_subagents=binding_contract.subagent_ids,
                ordinary_policy=ordinary_policy,
                mcp_policy=mcp_policy,
            )
            output_binding = restore_output(
                binding_contract.output_mode,
                binding_contract.output_schema,
            )
        except AIError as error:
            if error.code in {
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                ErrorCode.AGENT_BINDING_UNAVAILABLE,
            }:
                raise
            raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE) from error
        return AgentBinding(
            compiled_agent,
            output_binding,
            binding_contract,
        )

    def _restore_selected(
        self,
        pins: Sequence[CapabilityPin],
    ) -> "dict[str, tuple[CapabilityContribution[object], ...]]":
        selected: dict[str, list[CapabilityContribution[object]]] = {
            "tool": [],
            "skill": [],
            "mcp": [],
            "runtime_capability": [],
        }
        for pin in pins:
            if pin.kind == "skill":
                value = SkillDefinition.from_contract(
                    cast("Mapping[str, object]", pin.contract)
                )
                candidate = CapabilityContribution.from_declaration(value)
                if candidate.id != pin.id or candidate.revision != pin.revision:
                    raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
            elif pin.kind == "mcp":
                candidate = CapabilityContribution.from_mcp_contract(
                    pin.contract,
                )
            else:
                current = self._by_identity.get((pin.kind, pin.id))
                if (
                    current is None
                    or current.revision != pin.revision
                    or current.contract != dict(pin.contract)
                ):
                    raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
                candidate = current
            selected[pin.kind].append(candidate)
        return {
            kind: (
                tuple(values)
                if kind == "runtime_capability"
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
        select_all = "*" in spec.allow_tools
        selected_tool_ids: set[str] = set(tools) if select_all else set()
        selected_mcp_by_id: dict[str, CapabilityContribution[object]] = (
            dict(self._mcp_by_id) if select_all else {}
        )
        ordinary_policy: list[str] = ["*"] if select_all else []
        mcp_policy: list[str] = (
            [
                mcp_server_selector(item.id)
                for item in sorted(self._mcp_by_id.values(), key=lambda item: item.id)
            ]
            if select_all
            else []
        )
        workspace_tool_classes = {
            declaration.name: declaration.metadata["linktools.ai.tool_class"]
            for declaration in workspace_tool_declarations()
        }
        for selector in spec.allow_tools:
            if selector == "*":
                continue
            workspace_classes = _workspace_selector_classes(selector)
            if workspace_classes is not None:
                selected_workspace_tools = {
                    name
                    for name in tools
                    if workspace_tool_classes.get(name) in workspace_classes
                }
                selected_tool_ids.update(selected_workspace_tools)
                ordinary_policy.extend(selected_workspace_tools)
                continue
            parsed = parse_mcp_tool_selector(selector)
            if parsed is None:
                ordinary_policy.append(selector)
                if selector not in tools:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                selected_tool_ids.add(selector)
                continue
            server_id, _tool = parsed
            candidate = self._mcp_by_id.get(server_id)
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
        kind: Literal["skill", "runtime_capability"],
        selectors: Sequence[str],
    ) -> "tuple[CapabilityContribution[object], ...]":
        values = {
            candidate.id: candidate
            for candidate in self._candidates
            if candidate.kind == kind
        }
        select_all = "*" in selectors
        selected_ids = set(selectors).difference({"*"})
        if not selected_ids.issubset(values):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        selected = tuple(
            candidate
            for candidate in self._candidates
            if candidate.kind == kind
            and (select_all or candidate.id in selected_ids)
        )
        return (
            selected
            if kind == "runtime_capability"
            else tuple(sorted(selected, key=lambda item: item.id))
        )

    def _select_subagents(self, spec: AgentSpec) -> "tuple[str, ...]":
        available = set(self._agent_ids)
        select_all = "*" in spec.allow_subagents
        selected = set(spec.allow_subagents).difference({"*"})
        if spec.id in selected or not selected.issubset(available):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return tuple(
            sorted(
                available.difference({spec.id})
                if select_all
                else selected
            )
        )

    def _restore_policies(
        self,
        spec: AgentSpec,
        selected_tools: Sequence[CapabilityContribution[object]],
        selected_mcp: Sequence[CapabilityContribution[object]],
    ) -> "tuple[tuple[str, ...], tuple[str, ...]]":
        select_all = "*" in spec.allow_tools
        selected_tool_names = {item.id for item in selected_tools}
        ordinary: set[str] = {"*"} if select_all else set()
        workspace_tool_classes = {
            declaration.name: declaration.metadata["linktools.ai.tool_class"]
            for declaration in workspace_tool_declarations()
        }
        for selector in spec.allow_tools:
            if selector == "*":
                continue
            workspace_classes = _workspace_selector_classes(selector)
            if workspace_classes is not None:
                ordinary.update(
                    item.id
                    for item in selected_tools
                    if item.id in selected_tool_names
                    and workspace_tool_classes.get(item.id) in workspace_classes
                )
            elif parse_mcp_tool_selector(selector) is None:
                if selector not in selected_tool_names:
                    raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
                ordinary.add(selector)
        allowed_server_ids = {item.id for item in selected_mcp}
        mcp_policy = (
            [
                mcp_server_selector(item.id)
                for item in sorted(selected_mcp, key=lambda item: item.id)
            ]
            if select_all
            else []
        )
        for selector in spec.allow_tools:
            if selector == "*":
                continue
            parsed = parse_mcp_tool_selector(selector)
            if parsed is None:
                continue
            if parsed[0] not in allowed_server_ids:
                raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
            mcp_policy.append(selector)
        return tuple(sorted(ordinary)), tuple(sorted(set(mcp_policy)))

    def _build_compiled_agent(
        self,
        spec: AgentSpec,
        *,
        model: ModelBinding,
        selected_tools: Sequence[CapabilityContribution[object]],
        selected_skills: Sequence[CapabilityContribution[object]],
        selected_mcp: Sequence[CapabilityContribution[object]],
        selected_runtime_capabilities: Sequence[CapabilityContribution[object]],
        selected_subagents: Sequence[str],
        ordinary_policy: Sequence[str],
        mcp_policy: Sequence[str],
    ) -> CompiledAgent:
        selected_skill_ids = {candidate.id for candidate in selected_skills}
        if any(skill_id not in selected_skill_ids for skill_id in spec.preload_skills):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return CompiledAgent(
            spec=spec,
            model=model,
            selected_tools=tuple(sorted(selected_tools, key=lambda item: item.id)),
            selected_skills=tuple(sorted(selected_skills, key=lambda item: item.id)),
            selected_mcp=tuple(sorted(selected_mcp, key=lambda item: item.id)),
            selected_runtime_capabilities=tuple(selected_runtime_capabilities),
            selected_subagents=tuple(sorted(set(selected_subagents))),
            ordinary_tool_policy=tuple(ordinary_policy),
            mcp_selector_policy=tuple(mcp_policy),
        )


def _selected_candidates(
    compiled_agent: CompiledAgent,
) -> "tuple[CapabilityContribution[object], ...]":
    return tuple(
        (
            *sorted(
                (
                    *compiled_agent.selected_tools,
                    *compiled_agent.selected_skills,
                    *compiled_agent.selected_mcp,
                ),
                key=lambda item: (item.kind, item.id),
            ),
            *compiled_agent.selected_runtime_capabilities,
        )
    )


def _pin(candidate: CapabilityContribution[object]) -> CapabilityPin:
    return CapabilityPin(
        cast(Literal["tool", "skill", "mcp", "runtime_capability"], candidate.kind),
        candidate.id,
        candidate.contract,
    )


def _workspace_selector_classes(
    selector: str,
) -> "frozenset[str] | None":
    if selector == "file:*":
        return frozenset({"filesystem.read", "filesystem.write"})
    if selector == "file:read":
        return frozenset({"filesystem.read"})
    if selector == "file:write":
        return frozenset({"filesystem.write"})
    if selector == "terminal:*":
        return frozenset({"shell"})
    return None


__all__ = ["AgentCompiler"]
