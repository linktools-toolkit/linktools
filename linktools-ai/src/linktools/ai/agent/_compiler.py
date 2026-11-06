#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic Agent selection, binding, and exact historical recovery."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from inspect import iscoroutine
from types import MappingProxyType
from typing import Literal, TypeAlias

from pydantic import BaseModel

from ..capability import (
    CapabilityContribution,
    tool_class_from_metadata,
)
from ..core import ImmutableJsonMapping, JsonValue, validate_agent_id
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ._model_resolver import _ModelResolver
from ..spec import (
    DEFAULT_AGENT_RETRIES,
    AgentSpec,
    AgentSpecCodec,
    CapabilityRef,
    SubagentRef,
    mcp_server_name_token,
    mcp_server_selector,
    parse_mcp_tool_selector,
)
from ._binding import (
    AgentBinding,
    AgentBindingContract,
    _compiled_agent_selected_refs,
)
from ._compiled import CompiledAgent
from ._output import OutputBinding, bind_output, restore_output


CapabilityFilter: TypeAlias = Callable[[AgentSpec, CapabilityContribution[object]], bool]


@dataclass(frozen=True)
class _AgentDefaults:
    tool_retries: int = DEFAULT_AGENT_RETRIES
    output_retries: int = DEFAULT_AGENT_RETRIES

    def __post_init__(self) -> None:
        if self.tool_retries is None or self.output_retries is None:
            raise ValueError("agent retry defaults must be resolved")
        AgentSpec("defaults", tool_retries=self.tool_retries, output_retries=self.output_retries)


class AgentCompiler:
    """Own Agent selection within one current source observation."""

    def __init__(
        self,
        *,
        models: Mapping[str, ModelBinding | Callable[[], ModelBinding]],
        candidates: Sequence[CapabilityContribution[object]],
        agents: Mapping[str, AgentSpec],
        capability_filter: CapabilityFilter | None = None,
        defaults: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if models is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        settings = ImmutableJsonMapping({} if defaults is None else defaults)
        agent_settings = settings.get("agent", {})
        if not isinstance(agent_settings, dict):
            raise TypeError("agent defaults must be a JSON object")
        self._defaults = _AgentDefaults(**agent_settings)
        self._runtime_contributions = tuple(item for item in candidates if item.scope == "runtime")
        candidates = tuple(item for item in candidates if item.scope == "agent")
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
        for agent_id, specification in current_agents.items():
            if not isinstance(agent_id, str) or not isinstance(specification, AgentSpec) or specification.id != agent_id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if capability_filter is not None and not callable(capability_filter):
            raise TypeError("capability_filter must be callable or None")
        self._capability_filter = capability_filter
        self._models = _ModelResolver(models)
        self._candidates = ordered
        self._by_identity = {(item.kind, item.id): item for item in ordered}
        self._agents: Mapping[str, AgentSpec] = MappingProxyType(current_agents)
        server_tokens: dict[str, str] = {}
        for candidate in ordered:
            if candidate.kind == "tool" and candidate.id.startswith("mcp__"):
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            if candidate.kind != "mcp":
                continue
            token = mcp_server_name_token(candidate.id)
            previous = server_tokens.get(token)
            if previous is not None and previous != candidate.id:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            server_tokens[token] = candidate.id

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """Registered declarations in this resolution, without loading models."""
        return tuple(sorted(self._agents))

    def compile_registered(
        self,
        agent_id: str,
        derivations: Sequence[Mapping[str, object]] = (),
    ) -> CompiledAgent:
        """Compile one registered declaration and its ordered narrowing overlays."""
        validate_agent_id(agent_id)
        try:
            spec = self._agents[agent_id]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_BINDING_UNAVAILABLE,
                safe_details={"agent_id": agent_id},
            ) from error
        for changes in derivations:
            spec = self.derive_spec(spec, changes)
        return self.compile(spec)

    @classmethod
    def derive_spec(cls, spec: AgentSpec, changes: Mapping[str, object]) -> AgentSpec:
        """Narrow declared selectors without consulting an observed candidate set."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        if not set(changes).issubset({
            "model", "system_prompt", "instructions", "allow_tools", "allow_skills",
        }):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            derived = replace(spec, **changes)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        for allowed, requested in (
            (spec.allow_tools, derived.allow_tools),
            (spec.allow_skills, derived.allow_skills),
        ):
            if not all(_selector_is_contained(selector, allowed) for selector in requested):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return derived

    def compile(self, spec: AgentSpec) -> CompiledAgent:
        """Compile one declaration against this observation of available candidates."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        spec = replace(
            spec,
            tool_retries=self._defaults.tool_retries if spec.tool_retries is None else spec.tool_retries,
            output_retries=self._defaults.output_retries if spec.output_retries is None else spec.output_retries,
        )
        return self.compile_current(spec, model=self._models.resolve(spec.model))

    def compile_current(self, spec: AgentSpec, *, model: ModelBinding) -> CompiledAgent:
        """Select this observation using an already admitted model binding."""
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        declarations = tuple(item for item in self._candidates if item.kind != "agent")
        delegates = tuple(
            CapabilityContribution.from_declaration(agent) for agent in self._agents.values()
        )
        candidates = self.filter_capabilities(
            spec,
            (*declarations, *delegates),
            capability_filter=self._capability_filter,
        )
        selected_tools, selected_mcp, tool_policy, mcp_policy = self._select_tools(spec, candidates)
        selected_skills = self._select_exact_kind("skill", spec.allow_skills, candidates)
        selected_capabilities = self._select_exact_kind("capability", spec.allow_capabilities, candidates)
        selected_subagents = self._select_subagents(
            spec, tuple(item.id for item in candidates if item.kind == "agent"),
        )
        return self._build_compiled_agent(
            spec,
            model=model,
            selected_tools=selected_tools,
            selected_skills=selected_skills,
            selected_mcp=selected_mcp,
            selected_capabilities=selected_capabilities,
            selected_subagents=selected_subagents,
            tool_policy=tool_policy,
            mcp_policy=mcp_policy,
        )

    @classmethod
    def filter_capabilities(
        cls,
        spec: AgentSpec,
        contributions: Sequence[CapabilityContribution[object]],
        *,
        capability_filter: CapabilityFilter | None = None,
    ) -> tuple[CapabilityContribution[object], ...]:
        """Return visible candidates, not a fully resolved executable toolset.

        The optional synchronous filter runs before Agent selectors are resolved.
        It is not an authorization policy or a durable callback. Agent candidates
        are delegation targets; the root Agent declaration is never filtered.
        Runtime-scoped host policies are not Agent candidates and bypass this filter.
        """
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        if capability_filter is not None and not callable(capability_filter):
            raise TypeError("capability_filter must be callable or None")
        available: list[CapabilityContribution[object]] = []
        identities: set[tuple[str, str]] = set()
        for item in contributions:
            if not isinstance(item, CapabilityContribution):
                raise TypeError("contributions must contain CapabilityContribution values")
            identity = (item.kind, item.id)
            if identity in identities:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            identities.add(identity)
            if item.scope == "runtime" or (item.kind == "agent" and item.id == spec.id):
                continue
            if capability_filter is not None:
                included = capability_filter(spec, item)
                if not isinstance(included, bool):
                    if iscoroutine(included):
                        included.close()
                    raise TypeError("capability_filter must return bool synchronously")
                if not included:
                    continue
            available.append(item)
        return tuple(available)

    def bind(
        self,
        compiled_agent: CompiledAgent,
        *,
        output: "type[BaseModel] | OutputBinding | None" = None,
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
        output: "type[BaseModel] | OutputBinding | None" = None,
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
            selected_capabilities=compiled_agent.selected_capabilities,
            selected_subagents=(),
            tool_policy=compiled_agent.tool_policy,
            mcp_policy=compiled_agent.mcp_policy,
        )
        return self._bind(child_compiled_agent, output=output, subagents=())

    def _bind(
        self,
        compiled_agent: CompiledAgent,
        *,
        output: "type[BaseModel] | OutputBinding | None",
        subagents: Sequence[SubagentRef],
    ) -> AgentBinding:
        if isinstance(output, OutputBinding):
            output_binding = output
        elif output is None and compiled_agent.spec.output_schema is not None:
            output_binding = restore_output("structured", compiled_agent.spec.output_schema)
        else:
            output_binding = bind_output(output)
        # A current admission owns one binding, shared by contract and execution.
        compiled_agent = replace(
            compiled_agent,
            model=self._models.capture(self._models.resolve(compiled_agent.spec.model)),
        )
        binding_contract = AgentBindingContract(
            agent_spec=AgentSpecCodec().from_payload(
                AgentSpecCodec().to_wire_payload(compiled_agent.spec)
            ),
            model_contract=dict(compiled_agent.model.contract),
            selected=_compiled_agent_selected_refs(compiled_agent),
            subagents=tuple(subagents),
            output_mode=output_binding.mode,
            output_schema=output_binding.schema_definition,
        )
        return AgentBinding(
            compiled_agent,
            output_binding,
            binding_contract,
        )

    def restore(
        self, binding_contract: AgentBindingContract, *, model: ModelBinding | None = None,
    ) -> AgentBinding:
        """Restore admitted Agent facts and hooks with the current tool directory."""
        if not isinstance(binding_contract, AgentBindingContract):
            raise TypeError("binding_contract must be AgentBindingContract")
        try:
            if model is None:
                model = self._models.resolve(binding_contract.agent_spec.model)
            model = self._models.capture(model)
            selected_capabilities = self._restore_capabilities(binding_contract.selected)
            candidates = self.filter_capabilities(
                binding_contract.agent_spec,
                tuple(item for item in self._candidates if item.kind in {"tool", "skill", "mcp"}),
                capability_filter=self._capability_filter,
            )
            selected_tools, selected_mcp, tool_policy, mcp_policy = self._select_tools(
                binding_contract.agent_spec, candidates,
            )
            selected_skills = self._select_exact_kind(
                "skill", binding_contract.agent_spec.allow_skills, candidates,
            )
            compiled_agent = self._build_compiled_agent(
                binding_contract.agent_spec,
                model=model,
                selected_tools=selected_tools,
                selected_skills=selected_skills,
                selected_mcp=selected_mcp,
                selected_capabilities=selected_capabilities,
                selected_subagents=binding_contract.subagent_ids,
                tool_policy=tool_policy,
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

    def _restore_capabilities(
        self, refs: Sequence[CapabilityRef],
    ) -> tuple[CapabilityContribution[object], ...]:
        selected: list[CapabilityContribution[object]] = []
        for ref in refs:
            current = self._by_identity.get((ref.kind, ref.id))
            if ref.kind != "capability" or current is None or current.revision != ref.revision:
                raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
            selected.append(current)
        return tuple(selected)

    @classmethod
    def _select_tools(
        cls,
        spec: AgentSpec,
        candidates: Sequence[CapabilityContribution[object]],
    ) -> "tuple[tuple[CapabilityContribution[object], ...], tuple[CapabilityContribution[object], ...], tuple[str, ...], tuple[str, ...]]":
        tools = {
            candidate.id: candidate
            for candidate in candidates
            if candidate.kind == "tool"
        }
        mcp_by_id = {item.id: item for item in candidates if item.kind == "mcp"}
        select_all = "*" in spec.allow_tools
        selected_tool_ids: set[str] = set(tools) if select_all else set()
        selected_mcp_by_id: dict[str, CapabilityContribution[object]] = (
            dict(mcp_by_id) if select_all else {}
        )
        tool_policy: list[str] = ["*"] if select_all else []
        mcp_policy: list[str] = (
            [
                mcp_server_selector(item.id)
                for item in sorted(mcp_by_id.values(), key=lambda item: item.id)
            ]
            if select_all
            else []
        )
        workspace_tool_classes = {
            name: tool_class_from_metadata(candidate.contract.get("metadata"))
            for name, candidate in tools.items()
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
                tool_policy.extend(selected_workspace_tools)
                continue
            parsed = parse_mcp_tool_selector(selector)
            if parsed is None:
                tool_policy.append(selector)
                if selector in tools:
                    selected_tool_ids.add(selector)
                continue
            server_id, _tool = parsed
            candidate = mcp_by_id.get(server_id)
            if candidate is not None:
                selected_mcp_by_id[candidate.id] = candidate
            mcp_policy.append(selector)
        return (
            tuple(tools[name] for name in sorted(selected_tool_ids)),
            tuple(selected_mcp_by_id[name] for name in sorted(selected_mcp_by_id)),
            tuple(sorted(set(tool_policy))),
            tuple(sorted(set(mcp_policy))),
        )

    @classmethod
    def _select_exact_kind(
        cls,
        kind: Literal["skill", "capability"],
        selectors: Sequence[str],
        candidates: Sequence[CapabilityContribution[object]],
    ) -> "tuple[CapabilityContribution[object], ...]":
        select_all = "*" in selectors
        selected_ids = set(selectors).difference({"*"})
        selected = tuple(
            candidate
            for candidate in candidates
            if candidate.kind == kind
            and (select_all or candidate.id in selected_ids)
        )
        return (
            selected
            if kind == "capability"
            else tuple(sorted(selected, key=lambda item: item.id))
        )

    @classmethod
    def _select_subagents(cls, spec: AgentSpec, agent_ids: Sequence[str]) -> "tuple[str, ...]":
        available = set(agent_ids)
        select_all = "*" in spec.allow_subagents
        selected = set(spec.allow_subagents).difference({"*"})
        if spec.id in selected:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return tuple(
            sorted(
                available.difference({spec.id})
                if select_all
                else selected.intersection(available)
            )
        )

    def _build_compiled_agent(
        self,
        spec: AgentSpec,
        *,
        model: ModelBinding,
        selected_tools: Sequence[CapabilityContribution[object]],
        selected_skills: Sequence[CapabilityContribution[object]],
        selected_mcp: Sequence[CapabilityContribution[object]],
        selected_capabilities: Sequence[CapabilityContribution[object]],
        selected_subagents: Sequence[str],
        tool_policy: Sequence[str],
        mcp_policy: Sequence[str],
    ) -> CompiledAgent:
        return CompiledAgent(
            spec=spec,
            model=model,
            selected_tools=tuple(sorted(selected_tools, key=lambda item: item.id)),
            selected_skills=tuple(sorted(selected_skills, key=lambda item: item.id)),
            selected_mcp=tuple(sorted(selected_mcp, key=lambda item: item.id)),
            selected_capabilities=tuple(selected_capabilities),
            selected_subagents=tuple(sorted(set(selected_subagents))),
            tool_policy=tuple(tool_policy),
            mcp_policy=tuple(mcp_policy),
            runtime_contributions=self._runtime_contributions,
        )


def _selector_is_contained(selector: str, allowed: Sequence[str]) -> bool:
    if "*" in allowed or selector in allowed:
        return True
    parsed = parse_mcp_tool_selector(selector)
    if parsed is not None:
        return mcp_server_selector(parsed[0]) in allowed
    classes = _workspace_selector_classes(selector)
    if classes is None:
        return False
    allowed_classes = frozenset().union(*(
        _workspace_selector_classes(value) or frozenset() for value in allowed
    ))
    return classes.issubset(allowed_classes)


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


__all__ = ["AgentCompiler", "CapabilityFilter"]
