#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New admissions read current declarations; admitted bindings keep their facts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.agent import AgentBinding, AgentCatalog, AgentCompiler
from linktools.ai.capability import CapabilityContribution, SkillDefinition, tool_metadata
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionHandle
from linktools.ai.model import create_model
from linktools.ai.runtime._agent import Agent
from linktools.ai.runtime._agent_binding_resolver import _AgentBindingResolver
from linktools.ai.runtime.agent_execution import AgentExecutionPort
from linktools.ai.spec import AgentSpec, MCPServerSpec, SkillSpec


@dataclass
class _Extra(AbstractCapability[object]):
    id: str = "extra"


class _Answer(BaseModel):
    value: int


def _tool(name: str) -> CapabilityContribution[object]:
    def invoke() -> str:
        return name
    return CapabilityContribution.from_opaque(
        "tool", name, Tool(invoke, name=name, metadata=tool_metadata(effect_policy="none", tool_class="business")),
    )


def _compiler(
    specs: Sequence[AgentSpec], candidates: Sequence[CapabilityContribution[object]] = (),
    *, models: Mapping[str, object] | None = None,
) -> AgentCompiler:
    return AgentCompiler(
        models={"default": create_model(api="openai", model="test")} if models is None else models,
        candidates=candidates, agents={spec.id: spec for spec in specs},
    )


def _resolver(old: AgentCompiler, current: AgentCompiler) -> _AgentBindingResolver:
    async def provider() -> AgentCompiler:
        return current
    return _AgentBindingResolver(
        AgentCatalog({name: old.compile_registered(name) for name in old.agent_ids}),
        old, current_compiler=provider,
    )


class _Admissions:
    def __init__(self) -> None:
        self.binding: AgentBinding | None = None
        self.request = None
        self.session_id: str | None = None

    async def start(self, digest: str, request: object, **kwargs: object) -> ExecutionHandle:
        self.binding = kwargs["resolved_binding"]
        assert self.binding.binding_digest == digest
        self.request = request
        return ExecutionHandle("execution")

    async def resume(
        self, agent_id: str, digest: str, session_id: str, request: object, **kwargs: object,
    ) -> ExecutionHandle:
        self.session_id = session_id
        return await self.start(digest, request, **kwargs)


def _port(
    old: AgentCompiler, resolver: _AgentBindingResolver, admissions: _Admissions,
    *, agent_id: str = "root", captured: AgentBinding | None = None,
) -> AgentExecutionPort:
    compiled = old.compile_registered(agent_id) if agent_id in old.agent_ids else None
    return AgentExecutionPort(
        persistence=SimpleNamespace(tenant_id="tenant"), compiled=compiled,
        _binding_contract=None if captured is None else captured.binding_contract,
        compiler=old, service=admissions, sessions=admissions,
        operations=SimpleNamespace(ensure_open=lambda: None), watch_tree=None,
        resolve_binding=resolver.resolve, default_principal=Principal("user", "tenant"),
        correlation={}, captures=None, captured=captured is not None, agent_id=agent_id,
        resolve_current=resolver.resolve_agent, restore_binding=resolver.restore,
    )


async def _start(port: AgentExecutionPort, *, session_id: str | None = None) -> None:
    await port.start(
        "hello", files=(), output=None, principal=None, session_id=session_id,
        idempotency_key="call", mode="run", planning=None, thinking=None,
    )


@pytest.mark.asyncio
async def test_current_resolution_includes_every_capability_kind_and_new_agent() -> None:
    old = _compiler((AgentSpec("root", allow_subagents=()),))
    child = AgentSpec("child", revision=2, allow_subagents=())
    current = _compiler((AgentSpec("root", revision=2), child), (
        _tool("new_tool"),
        CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("guide", "Guide"))),
        CapabilityContribution.from_declaration(MCPServerSpec("server", "python")),
        CapabilityContribution.from_opaque("capability", "extra", _Extra()),
    ))
    resolver = _resolver(old, current)
    binding = await resolver.resolve_agent("root")
    assert binding.compiled_agent.spec.revision == 2
    assert {(pin.kind, pin.id) for pin in binding.binding_contract.selected} == {("capability", "extra")}
    assert binding.compiled_agent.static_tool_names == ("new_tool",)
    assert [item.id for item in binding.compiled_agent.selected_skills] == ["guide"]
    assert [item.id for item in binding.compiled_agent.selected_mcp] == ["server"]
    assert binding.binding_contract.subagent_bindings[0].agent_spec == current.compile_registered("child").spec
    assert (await resolver.resolve_agent("child")).compiled_agent.spec.revision == 2


@pytest.mark.asyncio
async def test_single_observation_resolves_root_and_children_without_unrelated_model_loads() -> None:
    old = _compiler((AgentSpec("root", allow_subagents=()),))
    loads: list[str] = []
    calls = 0

    def unused_model():
        loads.append("unused")
        raise AssertionError("Unrelated root model must not load")

    async def provider() -> AgentCompiler:
        nonlocal calls
        calls += 1
        return _compiler((
            AgentSpec("root", revision=calls, allow_subagents=("child",)),
            AgentSpec("child", revision=calls, allow_subagents=()),
            AgentSpec("unused", model="unused", allow_subagents=()),
        ), models={"default": create_model(api="openai", model="test"), "unused": unused_model})

    resolver = _AgentBindingResolver(AgentCatalog({}), old, current_compiler=provider)
    for expected in (1, 2):
        binding = await resolver.resolve_agent("root")
        assert binding.compiled_agent.spec.revision == expected
        assert binding.binding_contract.subagent_bindings[0].agent_spec.revision == expected
        assert calls == expected
    assert loads == []


@pytest.mark.parametrize(("allowed", "requested", "accepted"), (
    (("*",), ("future",), True),
    (("mcp:server:*",), ("mcp:server:future",), True),
    (("mcp:server:first",), ("mcp:server:second",), False),
    (("mcp:server:first",), ("mcp:server:*",), False),
    (("file:*",), ("file:read",), True),
    (("file:read",), ("file:write",), False),
    (("known",), ("future",), False),
))
def test_derivation_compares_declared_selector_ranges(allowed, requested, accepted) -> None:
    spec = AgentSpec("root", allow_tools=allowed)
    if accepted:
        assert AgentCompiler.derive_spec(spec, {"allow_tools": requested}).allow_tools == requested
    else:
        with pytest.raises(AIError) as raised:
            AgentCompiler.derive_spec(spec, {"allow_tools": requested})
        assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_derivation_keeps_future_names_but_revalidates_current_base_scope() -> None:
    old = _compiler((AgentSpec("root", allow_subagents=()),))
    current = _compiler((AgentSpec("root", allow_subagents=()),), (_tool("future"),))
    binding = await _resolver(old, current).resolve_agent("root", ({"allow_tools": ("future",)},))
    assert [item.id for item in binding.compiled_agent.selected_tools] == ["future"]
    narrowed = _compiler((AgentSpec("root", allow_tools=(), allow_subagents=()),), (_tool("future"),))
    with pytest.raises(AIError) as raised:
        await _resolver(old, narrowed).resolve_agent("root", ({"allow_tools": ("future",)},))
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", (None, "session"))
async def test_start_uses_current_definition_defaults_in_agent_and_session(session_id) -> None:
    old = _compiler((AgentSpec("root", revision=1, allow_subagents=()),))
    current = _compiler((AgentSpec("root", revision=2, planning=True, thinking="high",
                                    metadata={"label": "current"}, allow_subagents=()),), (_tool("new"),))
    admissions = _Admissions()
    await _start(_port(old, _resolver(old, current), admissions), session_id=session_id)
    assert admissions.binding.compiled_agent.spec.revision == 2
    assert admissions.binding.compiled_agent.spec.metadata == {"label": "current"}
    assert admissions.request.principal == Principal("user", "tenant")
    assert admissions.request.planning is True and admissions.request.thinking == "high"
    assert admissions.session_id == session_id


@pytest.mark.asyncio
async def test_restore_preserves_admitted_revision_output_selection_and_metadata() -> None:
    old = _compiler((AgentSpec("root", revision=1, metadata={"label": "admitted"},
                                allow_subagents=()),), (_tool("admitted"),))
    original = old.bind(old.compile_registered("root"), output=_Answer)
    current = _compiler((AgentSpec("root", revision=2, system_prompt="changed",
                                    metadata={"label": "new"}, allow_subagents=()),),
                        (_tool("admitted"), _tool("new")))
    resolver = _resolver(old, current)
    restored = await resolver.restore(original.binding_contract)
    assert restored.binding_contract == original.binding_contract
    assert restored.compiled_agent.spec.revision == 1
    assert restored.compiled_agent.spec.metadata == {"label": "admitted"}
    assert [item.id for item in restored.compiled_agent.selected_tools] == ["admitted", "new"]
    admissions = _Admissions()
    await _start(_port(old, resolver, admissions, captured=original))
    assert admissions.binding.binding_contract == original.binding_contract
    assert admissions.request.planning is False


@pytest.mark.asyncio
async def test_dependency_resolution_keeps_admitted_runtime_objects_and_output_validator() -> None:
    compiler = _compiler((AgentSpec("root", metadata={"label": "original"}, allow_subagents=("child",)),
                          AgentSpec("child", allow_subagents=())))
    binding = compiler.bind(compiler.compile_registered("root"), output=_Answer)
    resolved = await _resolver(compiler, compiler).resolve(binding)
    assert resolved.compiled_agent is binding.compiled_agent
    assert resolved.output_binding is binding.output_binding
    assert resolved.binding_contract.subagent_bindings


@pytest.mark.asyncio
async def test_unresolved_handle_is_honest_and_async_observation_does_not_pin_start() -> None:
    old = _compiler((AgentSpec("root", allow_subagents=()),))
    current = _compiler((AgentSpec("new", revision=2, allow_subagents=()),))

    async def provider() -> AgentCompiler:
        return current

    resolver = _AgentBindingResolver(AgentCatalog({}), old, current_compiler=provider)
    admissions = _Admissions()
    port = _port(old, resolver, admissions, agent_id="new")

    def execution_port(agent_id, revision, compiled, contract, *, derivations, observed_binding):
        return replace(port, compiled=compiled, _binding_contract=contract,
                       derivations=derivations, observed_binding=observed_binding)

    runtime = SimpleNamespace(_agent_execution_port=execution_port,
                              _compiled_agent=lambda agent_id, revision, compiled: compiled)
    handle = Agent(runtime, "new", None)
    assert handle.revision is None and handle.model is None
    with pytest.raises(AIError) as raised:
        handle.execution_port.binding_contract
    assert raised.value.safe_details["reason"] == "agent_definition_unresolved"
    observed = await handle.resolve()
    assert observed.revision == 2 and observed.model is not None
    assert observed.execution_port.binding_contract.agent_spec.revision == 2
    current = _compiler((AgentSpec("new", revision=3, allow_subagents=()),))
    await _start(observed.execution_port)
    assert observed.revision == 2
    assert admissions.binding.compiled_agent.spec.revision == 3
    await observed.session("new-session").start("continue", idempotency_key="continue")
    assert admissions.binding.compiled_agent.spec.revision == 3


@pytest.mark.asyncio
async def test_directory_changes_do_not_change_binding_identity_or_wire_contract() -> None:
    spec = AgentSpec("root", allow_subagents=())
    first = _compiler((spec,), (_tool("first"),))
    second = _compiler((spec,), (
        _tool("second"),
        CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("guide", "Guide"))),
        CapabilityContribution.from_declaration(MCPServerSpec("server", "python")),
    ))
    original = first.bind(first.compile_registered("root"))
    refreshed = second.bind(second.compile_registered("root"))
    assert original.binding_contract == refreshed.binding_contract
    assert original.binding_digest == refreshed.binding_digest
    assert original.binding_contract.selected == ()
    restored = await _resolver(first, second).restore(original.binding_contract)
    assert restored.compiled_agent.static_tool_names == ("second",)
    assert restored.binding_contract == original.binding_contract


def test_restore_filters_current_directory_but_preserves_admitted_run_hooks() -> None:
    first_hook = CapabilityContribution.from_opaque("capability", "extra", _Extra())
    first = _compiler((AgentSpec("root", allow_subagents=()),), (_tool("first"), first_hook))
    original = first.bind(first.compile_registered("root"))
    seen: list[tuple[str, str]] = []

    def include(spec: AgentSpec, candidate: CapabilityContribution[object]) -> bool:
        seen.append((candidate.kind, candidate.id))
        return candidate.id != "hidden"

    second = AgentCompiler(
        models={"default": create_model(api="openai", model="test")},
        agents={"root": AgentSpec("root", revision=3, allow_tools=(), allow_capabilities=())},
        candidates=(_tool("visible"), _tool("hidden"), first_hook,
                    CapabilityContribution.from_opaque("capability", "later", _Extra("later"))),
        capability_filter=include,
    )
    restored = second.restore(original.binding_contract)
    assert restored.compiled_agent.spec == original.compiled_agent.spec
    assert restored.compiled_agent.static_tool_names == ("visible",)
    assert [item.id for item in restored.compiled_agent.selected_capabilities] == ["extra"]
    assert set(seen) == {("tool", "visible"), ("tool", "hidden")}


def test_runtime_contributions_follow_observation_without_entering_durable_identity() -> None:
    spec = AgentSpec("root", allow_subagents=())
    first = _compiler((spec,))
    hook = CapabilityContribution.from_opaque("capability", "host", _Extra("host"), scope="runtime")
    second = _compiler((spec,), (hook,))
    original = first.bind(first.compile_registered("root"))
    current = second.bind(second.compile_registered("root"))
    restored = second.restore(original.binding_contract)
    assert original.binding_digest == current.binding_digest == restored.binding_digest
    assert restored.compiled_agent.runtime_contributions == (hook,)
    catalog = AgentCatalog({"root": original.compiled_agent})
    catalog.register_binding(original)
    assert catalog.register_binding(restored) is restored


def test_binding_rejects_directory_pins_as_admitted_run_hooks() -> None:
    from linktools.ai.spec import CapabilityRef
    compiler = _compiler((AgentSpec("root", allow_subagents=()),), (_tool("lookup"),))
    contract = compiler.bind(compiler.compile_registered("root")).binding_contract
    with pytest.raises(AIError) as raised:
        replace(contract, selected=(CapabilityRef("tool", "lookup", 1),))
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_task_invocation_keeps_declared_agent_and_current_directory() -> None:
    old = _compiler((AgentSpec("root", revision=1, allow_subagents=()),), (_tool("first"),))
    current = _compiler((AgentSpec("root", revision=2, planning=True, allow_subagents=()),), (_tool("new"),))
    admissions = _Admissions()
    port = _port(old, _resolver(old, current), admissions)
    declared = port.binding_contract
    await port.start("task", files=(), output=None, principal=None, session_id=None,
                     idempotency_key="task", mode="run", planning=None, thinking=None,
                     requires_task_invocation_capture=True)
    assert admissions.binding.binding_contract == declared
    assert admissions.binding.compiled_agent.spec.revision == 1
    assert admissions.binding.compiled_agent.static_tool_names == ("new",)
    assert admissions.request.planning is False


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", (False, True))
async def test_admission_memory_scope_comes_from_current_runtime_contributions(wrapped: bool) -> None:
    from pydantic_ai.capabilities import CombinedCapability, WrapperCapability
    from linktools.ai.capability import MemoryCapability
    old = _compiler((AgentSpec("root", allow_subagents=()),))
    memory = MemoryCapability(scope="current")
    capability = WrapperCapability(CombinedCapability([memory])) if wrapped else memory
    current = _compiler((AgentSpec("root", allow_subagents=()),), (
        CapabilityContribution.from_opaque("capability", capability.id or "wrapped-memory", capability, scope="runtime"),
    ))
    admissions = _Admissions()
    port = replace(_port(old, _resolver(old, current), admissions), memory_scope="stale")
    await _start(port)
    assert admissions.request.memory_scope == "current"
    removed = replace(_port(current, _resolver(current, old), admissions), memory_scope="stale")
    await _start(removed)
    assert admissions.request.memory_scope is None


def test_current_step_compilation_reuses_admitted_model_without_loading_route() -> None:
    def unexpected_model():
        raise AssertionError("Per-step resolution must not reload the model route")
    model = create_model(api="openai", model="admitted")
    compiler = _compiler((AgentSpec("root", allow_subagents=()),), (_tool("current"),),
                         models={"default": unexpected_model})
    compiled = compiler.compile_current(AgentSpec("root", allow_subagents=()), model=model)
    assert compiled.model is model
    assert compiled.static_tool_names == ("current",)


@pytest.mark.asyncio
async def test_recovery_declaration_comparison_ignores_display_metadata_not_behavior() -> None:
    first = _compiler((AgentSpec("root", allow_subagents=("child",)),
                       AgentSpec("child", allow_subagents=())))
    changed = _compiler((AgentSpec("root", allow_subagents=("child",)),
                         AgentSpec("child", description="new", metadata={"label": "new"}, allow_subagents=())))
    original = (await _resolver(first, first).resolve_agent("root")).binding_contract
    observed = (await _resolver(first, changed).resolve_agent("root")).binding_contract
    assert original != observed
    assert original.matches_declaration(observed)
    assert original.subagent_bindings[0].agent_spec.metadata == {}
    changed_behavior = replace(observed.subagent_bindings[0],
                               agent_spec=replace(observed.subagent_bindings[0].agent_spec, system_prompt="changed"))
    assert not original.matches_declaration(replace(observed, subagent_bindings=(changed_behavior,)))


def test_allowed_selectors_remain_valid_when_resources_are_absent() -> None:
    spec = AgentSpec("root", allow_tools=("future", "mcp:future:*"), allow_skills=("future",),
                     allow_capabilities=("future",), allow_subagents=("future",))
    compiled = _compiler((spec,)).compile_registered("root")
    assert compiled.static_tool_names == ()
    assert compiled.selected_skills == compiled.selected_mcp == compiled.selected_capabilities == ()
    assert compiled.selected_subagents == ()
    redundant = _compiler((replace(spec, allow_tools=("*", "future")),)).compile_registered("root")
    assert redundant.static_tool_names == ()


@pytest.mark.asyncio
async def test_task_child_resolution_cannot_replace_declared_named_revision() -> None:
    old = _compiler((AgentSpec("root", allow_subagents=("child",)), AgentSpec("child", allow_subagents=())))
    current = _compiler((AgentSpec("root", allow_subagents=("child",)),
                         AgentSpec("child", revision=2, allow_subagents=())))
    declared = old.bind(old.compile_registered("root")).binding_contract
    with pytest.raises(AIError) as raised:
        await _resolver(old, current).restore(declared)
    assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
    original = (await _resolver(old, old).resolve_agent("root")).binding_contract
    with pytest.raises(AIError) as raised:
        replace(original, subagent_bindings=(replace(
            original.subagent_bindings[0], agent_spec=replace(original.subagent_bindings[0].agent_spec, revision=2),
        ),))
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
